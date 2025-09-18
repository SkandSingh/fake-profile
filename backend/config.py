"""
Configuration file for ML models and training
"""

# Model configurations
MODEL_CONFIGS = {
    "efficientnet_b0": {
        "model_type": "timm",
        "model_name": "efficientnet_b0",
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "epochs": 30
    },
    "resnet18": {
        "model_type": "timm", 
        "model_name": "resnet18",
        "batch_size": 256,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "epochs": 25
    },
    "vit_base": {
        "model_type": "huggingface",
        "model_name": "google/vit-base-patch16-224",
        "batch_size": 64,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "epochs": 40
    }
}

# Default training configuration
DEFAULT_CONFIG = {
    "seed": 42,
    "use_wandb": False,
    "project_name": "image-classification"
}

# CIFAR-10 class information
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info"
}