# utils.py
import hashlib
import os
import dataclasses
from dataclasses import dataclass
import torch
from transformer.modelling.functional.Transformer import Transformer


@dataclass
class TrainingConfig:
    # Model architecture parameters (affect hash)
    vocab_size: int = 50000
    d_model: int = 64
    n_heads: int = 4
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_len: int = 128

    # Training parameters (don't affect hash)
    dataset_portion: float = 0.4
    validation_portion: float = 0.3
    batch_size: int = 64
    num_epochs: int = 5
    lr: float = 1e-3
    warmup_portion: float = 0.3
    weight_decay: float = 0.01
    save_dir: str = "checkpoints"


def get_config_hash(config: TrainingConfig) -> str:
    """Create hash from model architecture parameters"""
    hash_params = {
        'vocab_size': config.vocab_size,
        'd_model': config.d_model,
        'n_heads': config.n_heads,
        'num_encoder_layers': config.num_encoder_layers,
        'num_decoder_layers': config.num_decoder_layers,
        'dim_feedforward': config.dim_feedforward,
        'dropout': config.dropout,
        'max_len': config.max_len,
    }
    return hashlib.md5(str(hash_params).encode()).hexdigest()[:8]


def get_model_path(config: TrainingConfig) -> str:
    """Get model path with hash, create directory if needed"""
    os.makedirs(config.save_dir, exist_ok=True)
    config_hash = get_config_hash(config)
    return os.path.join(config.save_dir, f"best_model_{config_hash}.pth")


def load_or_create_model(config: TrainingConfig, device="cpu") -> Transformer:
    """Load existing model or create new one with config"""
    model_path = get_model_path(config)

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = Transformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_len=config.max_len,
            device=device
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    print("Creating new model")
    return Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_len=config.max_len,
        device=device
    )