"""
ML Model Training Script with Reproducible Seeds
Production-ready image classification training with PyTorch and HuggingFace
"""
# mypy: ignore-errors
# type: ignore
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import timm
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, classification_report
import wandb
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Set reproducible seeds
def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class ImageClassificationTrainer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        set_seed(config['seed'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[Any] = None
        self.train_loader: Optional[Any] = None
        self.val_loader: Optional[Any] = None
        self.test_loader: Optional[Any] = None
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(project=config['project_name'], config=config)
    
    def prepare_data(self):
        """Prepare CIFAR-10 dataset with transforms"""
        # Data transforms
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], 
                                     shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], 
                                   shuffle=False, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], 
                                    shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
    
    def create_model(self):
        """Create model based on config"""
        if self.config['model_type'] == 'timm':
            self.model = timm.create_model(
                self.config['model_name'], 
                pretrained=True, 
                num_classes=10
            )
        elif self.config['model_type'] == 'huggingface':
            self.model = AutoModelForImageClassification.from_pretrained(
                self.config['model_name'],
                num_labels=10,
                ignore_mismatched_sizes=True
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config['model_type']}")
        
        self.model = self.model.to(self.device)
        print(f"Model created: {self.config['model_name']}")
        if self.model is not None:
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, optimizer: Any, criterion: Optional[Any]) -> Tuple[float, float]:
        """Train for one epoch"""
        if self.model is None or self.train_loader is None:
            raise ValueError("Model and train_loader must be initialized")
            
        self.model.train()  # type: ignore
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')  # type: ignore
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            if self.config['model_type'] == 'huggingface':
                outputs = self.model(pixel_values=data, labels=target)
                loss = outputs.loss
                predictions = outputs.logits.argmax(dim=-1)
            else:
                output = self.model(data)
                loss = criterion(output, target)
                predictions = output.argmax(dim=1)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += predictions.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, criterion):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.config['model_type'] == 'huggingface':
                    outputs = self.model(pixel_values=data, labels=target)
                    loss = outputs.loss
                    predictions = outputs.logits.argmax(dim=-1)
                else:
                    output = self.model(data)
                    loss = criterion(output, target)
                    predictions = output.argmax(dim=1)
                
                total_loss += loss.item()
                correct += predictions.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def test(self):
        """Test model on test set"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.config['model_type'] == 'huggingface':
                    outputs = self.model(pixel_values=data)
                    pred = outputs.logits.argmax(dim=-1)
                else:
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(targets, predictions)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        report = classification_report(targets, predictions, target_names=class_names)
        print("\nClassification Report:")
        print(report)
        
        return accuracy, report
    
    def save_model(self, filename=None):
        """Save trained model"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_{self.config['model_name'].replace('/', '_')}_{timestamp}.pth"
        
        model_dir = Path("./models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / filename
        
        # Save model state and config
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_name': self.config['model_name'],
            'model_type': self.config['model_type']
        }
        
        torch.save(save_dict, model_path)
        print(f"Model saved to: {model_path}")
        return model_path
    
    def train(self):
        """Main training loop"""
        self.prepare_data()
        self.create_model()
        
        # Define loss and optimizer
        if self.config['model_type'] == 'huggingface':
            # HuggingFace models have built-in loss
            criterion = None
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.5, verbose=True
        )
        
        best_val_acc = 0
        best_model_path = None
        
        print(f"Starting training for {self.config['epochs']} epochs")
        print(f"Device: {self.device}")
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = self.save_model(f"best_model_{self.config['model_name'].replace('/', '_')}.pth")
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
        
        # Test best model
        if best_model_path:
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            test_acc, test_report = self.test()
            
            if self.config.get('use_wandb', False):
                wandb.log({'test_accuracy': test_acc})
                wandb.finish()
        
        return best_model_path, best_val_acc, test_acc


def main():
    """Main function to run training"""
    # Default configuration
    config = {
        'model_type': 'timm',  # 'timm' or 'huggingface'
        'model_name': 'efficientnet_b0',  # or 'google/vit-base-patch16-224'
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'epochs': 20,
        'seed': 42,
        'use_wandb': False,
        'project_name': 'image-classification'
    }
    
    # Initialize trainer
    trainer = ImageClassificationTrainer(config)
    
    # Train model
    model_path, val_acc, test_acc = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()