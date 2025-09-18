"""
Simple test script to validate model training functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.train import ImageClassificationTrainer, set_seed
import torch

def test_training():
    """Test basic training functionality"""
    print("Testing ML training pipeline...")
    
    # Minimal config for quick testing
    config = {
        'model_type': 'timm',
        'model_name': 'efficientnet_b0',
        'batch_size': 16,  # Small batch for testing
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'epochs': 1,  # Just one epoch for testing
        'seed': 42,
        'use_wandb': False,
        'project_name': 'test-run'
    }
    
    try:
        # Initialize trainer
        trainer = ImageClassificationTrainer(config)
        
        # Test data preparation
        trainer.prepare_data()
        print("✅ Data preparation successful")
        
        # Test model creation
        trainer.create_model()
        print("✅ Model creation successful")
        
        # Check if we can do a forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        trainer.model.eval()
        with torch.no_grad():
            output = trainer.model(dummy_input)
            assert output.shape[1] == 10, "Output should have 10 classes"
        
        print("✅ Model forward pass successful")
        print(f"✅ All tests passed! Model ready for training on {device}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_training()
    sys.exit(0 if success else 1)