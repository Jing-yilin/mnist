import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from scripts.download_data import download_mnist_dataset
import os
import json
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

# Fix random seeds for reproducibility
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    """CNN model for MNIST classification"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # First convolutional layer: 1 channel in, 32 out, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # Second convolutional layer: 32 channels in, 64 out, 3x3 kernel
        self.dropout1 = nn.Dropout(0.25)  # Dropout layer with 0.25 probability
        self.dropout2 = nn.Dropout(0.5)   # Dropout layer with 0.5 probability
        self.fc1 = nn.Linear(9216, 128)   # First fully connected layer
        self.fc2 = nn.Linear(128, 10)     # Output layer: 10 classes for digits 0-9

    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)  # Apply log softmax for NLL loss
        return output


def train(args: argparse.Namespace, model: nn.Module, device: torch.device, 
         train_loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, 
         epoch: int, results: Dict[str, Any]) -> None:
    """Train the model for one epoch"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear gradients
        output = model(data)   # Forward pass
        loss = F.nll_loss(output, target)  # Calculate negative log likelihood loss
        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # Store training loss
            results["training_history"].append({
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss.item(),
                "step": (epoch - 1) * len(train_loader) + batch_idx
            })
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", loss.item(), 
                            step=(epoch - 1) * len(train_loader) + batch_idx)
            if args.dry_run:
                break


def test(model: nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader, 
        results: Dict[str, Any], epoch: Optional[int] = None) -> Dict[str, Any]:
    """Evaluate model on test data"""
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    with torch.no_grad():  # Disable gradient computation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    # Store test results
    test_result = {
        "loss": test_loss,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_loader.dataset)
    }
    
    if epoch is not None:
        test_result["epoch"] = epoch
        results["testing_history"].append(test_result)
        # Log test metrics to MLflow
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_accuracy", accuracy, step=epoch)
    
    results["final_test_result"] = test_result
    return test_result


def save_results(args: argparse.Namespace, results: Dict[str, Any]) -> str:
    """Save training results to a JSON file"""
    os.makedirs("results", exist_ok=True)
    filename = f"results/mnist_results.json"
    
    # Add training parameters to results
    results["parameters"] = {
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "seed": args.seed
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 [RESULTS] Training results saved to {filename}")
    return filename


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='directory containing the dataset (default: ./data)')
    parser.add_argument('--mlflow-tracking-uri', type=str, default='./mlruns',
                        help='MLflow tracking URI (default: ./mlruns)')
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    # Start MLflow run
    with mlflow.start_run():
        # Determine device to use (CUDA, MPS, or CPU)
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        use_mps = not args.no_mps and torch.backends.mps.is_available()

        if use_cuda:
            device = torch.device("cuda")
        elif use_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        # Log parameters
        mlflow.log_params({
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "seed": args.seed,
            "device": str(device)
        })
        
        # Set all random seeds for reproducibility
        set_seed(args.seed)

        # Configure data loaders
        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        # Load MNIST dataset from data directory
        print(f"📂 [DATA] Loading MNIST dataset from {args.data_dir}...")
        train_dataset, test_dataset = download_mnist_dataset(args.data_dir)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        
        # Initialize results dictionary to track training progress
        results = {
            "training_history": [],
            "testing_history": [],
            "final_test_result": None,
        }
        
        # Initialize model, optimizer, and learning rate scheduler
        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        
        # Log model architecture as a string instead of trying to serialize tensors
        model_architecture = {
            'name': model.__class__.__name__,
            'layers': {
                'conv1': f"Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))",
                'conv2': f"Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))",
                'dropout1': "Dropout(p=0.25)",
                'dropout2': "Dropout(p=0.5)",
                'fc1': "Linear(in_features=9216, out_features=128)",
                'fc2': "Linear(in_features=128, out_features=10)"
            }
        }
        mlflow.log_dict(model_architecture, "model_architecture.json")
        
        # Training loop
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, results)
            test(model, device, test_loader, results, epoch)
            scheduler.step()

        # Run final test
        print(f"🔍 [EVALUATE] Running final test...")
        final_result = test(model, device, test_loader, results)
        
        # Save model if requested
        model_path = None
        if args.save_model:
            os.makedirs("models", exist_ok=True)
            model_path = f"models/mnist_cnn.pt"
            torch.save(model.state_dict(), model_path)
            print(f"💾 [MODEL] Model saved to {model_path}")
            
            # Log model to MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model"
            )
        
        # Save results and print final summary
        results_file = save_results(args, results)
        print(f"🎉 [COMPLETE] Training completed, accuracy: {final_result['accuracy']:.2f}%")
        
        # Log final metrics
        mlflow.log_metrics({
            "final_test_loss": final_result["loss"],
            "final_test_accuracy": final_result["accuracy"]
        })


if __name__ == '__main__':
    main()
