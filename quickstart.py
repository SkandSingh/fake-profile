"""
Quick start script for training a model and starting the API
"""
import os
import sys
import subprocess
from pathlib import Path

def quick_start():
    """Quick start the ML application"""
    print("🚀 Quick Start - ML Application")
    print("="*50)
    
    # Check if we're in the right directory
    if not Path("backend").exists() or not Path("frontend").exists():
        print("❌ Please run this script from the project root directory")
        return
    
    # Check Python version
    try:
        import sys
        if sys.version_info < (3, 11):
            print(f"⚠️  Python 3.11+ recommended. Current: {sys.version}")
    except:
        pass
    
    print("1️⃣  Training a quick model (1 epoch for demo)...")
    
    # Quick training
    try:
        os.chdir("backend/ml")
        
        # Run quick training
        result = subprocess.run([
            sys.executable, "-c", """
from train import ImageClassificationTrainer

config = {
    'model_type': 'timm',
    'model_name': 'efficientnet_b0',
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'epochs': 1,  # Quick demo
    'seed': 42,
    'use_wandb': False,
    'project_name': 'quickstart'
}

trainer = ImageClassificationTrainer(config)
model_path, val_acc, test_acc = trainer.train()
print(f"Model saved to: {model_path}")
print(f"Validation accuracy: {val_acc:.2f}%")
"""
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model training completed!")
            print(result.stdout)
        else:
            print("⚠️  Training skipped (dependencies may not be installed)")
            print("Run: pip install -r backend/requirements.txt")
    
    except Exception as e:
        print(f"⚠️  Training skipped: {e}")
    
    finally:
        os.chdir("../..")  # Back to project root
    
    print("\n2️⃣  Starting API server...")
    print("Run this command in a new terminal:")
    print("cd backend && python -m api.main")
    
    print("\n3️⃣  Starting frontend...")
    print("Run this command in another terminal:")
    print("cd frontend && npm install && npm run dev")
    
    print("\n🎉 Quick start complete!")
    print("Visit http://localhost:3000 once both servers are running")

if __name__ == "__main__":
    quick_start()