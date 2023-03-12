import torch
import json





def prepare_optimizer_parameters(args, model):
    config = args.deepspeed_config
    deepspeed_config = json.load(
        open(args.deepspeed_config, 'r', encoding='utf-8'))

    param_optimizer = list(model.network.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.deepspeed_transformer_kernel:
        no_decay = no_decay + [
            'attn_nw', 'attn_nb', 'norm_w', 'norm_b', 'attn_qkvb', 'attn_ob',
            'inter_b', 'output_b'
        ]
    if "weight_decay" in config["training"].keys():
        weight_decay = config["training"]["weight_decay"]
    else:
        weight_decay = 0.01

    if deepspeed_config["optimizer"]["type"] not in [
            "OneBitAdam", "OneBitLamb", "ZeroOneAdam"
    ]:
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            weight_decay
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
    else:
        # Because 1-bit compression cannot represent exact zero, it is required to
        # provide a momentum mask for those params that have constant exact zeros in their
        # momentums, otherwise the compression error would keep accumulating.
        # For example, for bert pre-training seq 128, bert.embeddings.position_embeddings.weight
        # always have exact zeros in its momentum for row 129 to 512, because it only
        # learns up to seq length 128 while the model supports up to 512 seq length.
        need_mask = ['position_embeddings.weight']
        need_mask_p = []
        need_mask_decay = []
        masks = []
        for n, p in param_optimizer:
            if any(nd in n for nd in need_mask):
                mask = torch.zeros_like(p.data)
                for position in range(args.max_seq_length):
                    for col in range(p.size()[1]):
                        mask[position][col] += 1
                if deepspeed_config["optimizer"]["type"] in ["OneBitAdam", "ZeroOneAdam"]:
                    mask = torch.flatten(mask)
                masks.append(mask)
                need_mask_p.append(p)
                if any(nd in n for nd in no_decay):
                    need_mask_decay.append(0.0)
                else:
                    need_mask_decay.append(weight_decay)

        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay + need_mask)
            ],
            'weight_decay':
            weight_decay
        }, {
            'params': [
                p for n, p in param_optimizer
                if (any(nd in n
                        for nd in no_decay) and not any(nd in n
                                                        for nd in need_mask))
            ],
            'weight_decay':
            0.0
        }]

        for i_mask in range(len(need_mask_p)):
            optimizer_grouped_parameters.append({
                'params': [need_mask_p[i_mask]],
                'weight_decay':
                need_mask_decay[i_mask],
                'exp_avg_mask':
                masks[i_mask]
            })

    return optimizer_grouped_parameters