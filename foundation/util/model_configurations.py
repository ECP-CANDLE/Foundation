def setup_gpt_configuration(name: str) -> Dict[str: int]:
    """Set up the configuration for the GPT model."""
    if name == "gpt2":
        return {"n_embd": 768, "n_layer": 12, "n_head": 12}
    elif name == "gpt2-medium":
        return {"n_embd": 1024, "n_layer": 24, "n_head": 16}
    elif name == "gpt2-large":
        return {"n_embd": 1280, "n_layer": 36, "n_head": 20}
    elif name == "gpt2-xl":
        return {"n_embd": 1600, "n_layer": 48, "n_head": 25}
    elif name == "gpt3":
        return {"n_embd": 12288, "n_layer": 96, "n_head": 96}
    elif name == "gpt-1T":
        return {"n_embd": 4096, "n_layer": 32, "n_head": 32}
    else:
        raise ValueError(f"Unknown GPT model name: {name}")