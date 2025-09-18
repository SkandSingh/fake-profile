"""
Hyperparameter Optimization with Optuna
Production-ready hyperparameter tuning for ML models
"""
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import timm
import numpy as np
from transformers import AutoModelForImageClassification
from sklearn.metrics import accuracy_score
import json
import os
from datetime import datetime
from pathlib import Path
import logging
from tqdm import tqdm

# Import our training module
from train import set_seed, ImageClassificationTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    def __init__(self, base_config, n_trials=50, study_name=None):
        self.base_config = base_config
        self.n_trials = n_trials
        self.study_name = study_name or f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.best_params = None
        self.best_score = 0
        
        # Create study directory
        self.study_dir = Path("./studies")
        self.study_dir.mkdir(exist_ok=True)
        
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        # Set seed for reproducibility
        set_seed(self.base_config['seed'])
        
        # Suggest hyperparameters
        config = self.base_config.copy()
        
        # Model selection
        if config.get('optimize_model', True):
            model_choices = [
                'efficientnet_b0', 'efficientnet_b1', 'resnet18', 'resnet34',
                'mobilenetv3_small_100', 'mobilenetv3_large_100'
            ]
            config['model_name'] = trial.suggest_categorical('model_name', model_choices)
        
        # Learning rate optimization
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # Batch size optimization
        config['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        
        # Weight decay optimization
        config['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
        
        # Optimizer selection
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
        config['optimizer'] = optimizer_name
        
        # Dropout rate (if applicable)
        config['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        # Data augmentation strength
        config['augmentation_strength'] = trial.suggest_float('augmentation_strength', 0.1, 0.8)
        
        # Reduce epochs for faster optimization
        config['epochs'] = min(10, config.get('epochs', 20))
        
        try:
            # Create trainer with optimized config
            trainer = OptimizedImageClassificationTrainer(config, trial)
            
            # Train model and get validation accuracy
            val_accuracy = trainer.train_with_pruning()
            
            # Report intermediate values for pruning
            trial.report(val_accuracy, step=config['epochs'])
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return val_accuracy
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return 0.0  # Return poor score for failed trials
    
    def optimize(self):
        """Run hyperparameter optimization"""
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=self.study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        logger.info(f"Starting optimization with {self.n_trials} trials")
        
        # Optimize
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Save results
        self.save_results(study)
        
        # Print results
        self.print_results(study)
        
        return study
    
    def save_results(self, study):
        """Save optimization results"""
        results = {
            'study_name': self.study_name,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'datetime': datetime.now().isoformat()
        }
        
        # Save detailed results
        results_path = self.study_dir / f"{self.study_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save study object
        study_path = self.study_dir / f"{self.study_name}_study.pkl"
        with open(study_path, 'wb') as f:
            import pickle
            pickle.dump(study, f)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Study saved to {study_path}")
    
    def print_results(self, study):
        """Print optimization results"""
        print("\n" + "="*50)
        print("HYPERPARAMETER OPTIMIZATION RESULTS")
        print("="*50)
        print(f"Study Name: {self.study_name}")
        print(f"Number of trials: {len(study.trials)}")
        print(f"Best validation accuracy: {study.best_value:.4f}")
        print("\nBest parameters:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
        
        # Print parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            print("\nParameter importance:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {param}: {imp:.4f}")
        except:
            pass
        
        print("="*50)
    
    def train_best_model(self, full_epochs=None):
        """Train final model with best parameters"""
        if self.best_params is None:
            raise ValueError("No optimization results found. Run optimize() first.")
        
        # Create config with best parameters
        config = self.base_config.copy()
        config.update(self.best_params)
        
        if full_epochs:
            config['epochs'] = full_epochs
        
        logger.info("Training final model with best parameters")
        trainer = ImageClassificationTrainer(config)
        model_path, val_acc, test_acc = trainer.train()
        
        return model_path, val_acc, test_acc


class OptimizedImageClassificationTrainer(ImageClassificationTrainer):
    """Extended trainer for optimization with pruning support"""
    
    def __init__(self, config, trial=None):
        super().__init__(config)
        self.trial = trial
    
    def create_model(self):
        """Create model with potential dropout modifications"""
        super().create_model()
        
        # Add dropout if specified
        if hasattr(self.model, 'classifier') and self.config.get('dropout_rate'):
            # For timm models with classifier
            if isinstance(self.model.classifier, nn.Linear):
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(self.config['dropout_rate']),
                    nn.Linear(in_features, 10)
                )
    
    def get_optimizer(self):
        """Get optimizer based on config"""
        optimizer_name = self.config.get('optimizer', 'adamw')
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def get_augmented_transforms(self):
        """Get data transforms with configurable augmentation strength"""
        strength = self.config.get('augmentation_strength', 0.3)
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(int(10 * strength)),
            transforms.ColorJitter(
                brightness=0.2 * strength,
                contrast=0.2 * strength,
                saturation=0.2 * strength,
                hue=0.1 * strength
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform
    
    def train_with_pruning(self):
        """Training loop with Optuna pruning support"""
        self.prepare_data_optimized()
        self.create_model()
        
        # Get optimizer
        optimizer = self.get_optimizer()
        
        # Define loss
        if self.config['model_type'] == 'huggingface':
            criterion = None
        else:
            criterion = nn.CrossEntropyLoss()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=2, factor=0.5
        )
        
        best_val_acc = 0
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Update best accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Report to Optuna for pruning
            if self.trial:
                self.trial.report(val_acc, step=epoch)
                
                # Check if trial should be pruned
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
        
        return best_val_acc
    
    def prepare_data_optimized(self):
        """Prepare data with optimized transforms"""
        # Get augmented transforms
        train_transform = self.get_augmented_transforms()
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=val_transform)
        
        # Split train into train/val
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config['seed'])
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,  # Reduced for optimization
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )


def main():
    """Main function for hyperparameter optimization"""
    # Base configuration
    base_config = {
        'model_type': 'timm',
        'seed': 42,
        'use_wandb': False,
        'project_name': 'hyperparam-optimization',
        'optimize_model': True  # Whether to optimize model architecture
    }
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        n_trials=20,  # Adjust based on computational budget
        study_name="cifar10_optimization"
    )
    
    # Run optimization
    study = optimizer.optimize()
    
    # Train best model with full epochs
    print("\nTraining final model with best parameters...")
    model_path, val_acc, test_acc = optimizer.train_best_model(full_epochs=50)
    
    print(f"\nFinal Results:")
    print(f"Model saved to: {model_path}")
    print(f"Validation accuracy: {val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()