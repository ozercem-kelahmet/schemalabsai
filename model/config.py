DEFAULT_CONFIG = {
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'schema_layers': 2,
    'n_latents': 64,
    'vocab_size': 50000,
    'n_types': 10,
    'n_domains': 1000,
    'n_classes': 10,
    'max_cols': 64,
    'n_features': 10,
    'sector': 'default',
    'batch_size': 256,
    'learning_rate': 1e-4,
    'max_epochs': 5
}

def get_config(overrides=None):
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config
