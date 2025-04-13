import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import random
import numpy as np
from typing import Dict, Any
from scripts.download_data import download_mnist_dataset
import mlflow
import mlflow.pytorch

# Import the model definition from train.py
from train import Net, set_seed


def test(model: nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
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
    
    # Return test results
    return {
        "loss": test_loss,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_loader.dataset)
    }


def save_test_results(args: argparse.Namespace, test_result: Dict[str, Any]) -> str:
    """Save test results to a JSON file"""
    # Get environment type from PIXI_ENVIRONMENT_NAME environment variable
    env_type = os.environ.get("PIXI_ENVIRONMENT_NAME", "default")
    is_cuda = torch.cuda.is_available()
    
    # Create environment-specific directory
    result_dir = os.path.join("results", env_type)
    os.makedirs(result_dir, exist_ok=True)
    filename = os.path.join(result_dir, "mnist_test_results.json")
    
    # Create results dictionary with test parameters
    results = {
        "test_result": test_result,
        "parameters": {
            "test_batch_size": args.test_batch_size,
            "seed": args.seed,
            "environment": env_type,
            "cuda_available": is_cuda,
            "model_path": args.model_path
        }
    }
    
    # Add additional CUDA information if available
    if is_cuda:
        results["cuda_info"] = {
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, "version") else "N/A",
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count()
        }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 [RESULTS] Test results saved to {filename}")
    return filename


def main():
    # Testing settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Model Evaluation')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA evaluation')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU evaluation')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='path to the trained model to evaluate')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='directory containing the dataset (default: ./data)')
    parser.add_argument('--mlflow-tracking-uri', type=str, default='./mlruns',
                        help='MLflow tracking URI (default: ./mlruns)')
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    # Get environment type from PIXI_ENVIRONMENT_NAME environment variable
    env_type = os.environ.get("PIXI_ENVIRONMENT_NAME", "default")
    
    # Start MLflow run with environment name in the run name
    with mlflow.start_run(run_name=f"mnist_evaluation_{env_type}"):
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
            "test_batch_size": args.test_batch_size,
            "seed": args.seed,
            "device": str(device),
            "environment": env_type,
            "model_path": args.model_path
        })
        
        # Set all random seeds for reproducibility
        set_seed(args.seed)

        # Configure data loader
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': False}  # No need to shuffle test data
            test_kwargs.update(cuda_kwargs)

        # Load MNIST dataset from data directory (only need test dataset)
        print(f"📂 [DATA] Loading MNIST dataset from {args.data_dir}...")
        _, test_dataset = download_mnist_dataset(args.data_dir)
        
        # Create test data loader
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
        # Load model
        print(f"📦 [MODEL] Loading model from {args.model_path}...")
        model = Net().to(device)
        model.load_state_dict(torch.load(args.model_path))
        
        # Run evaluation
        print(f"🔍 [EVALUATE] Evaluating model on test data...")
        test_result = test(model, device, test_loader)
        
        # Save and log results
        results_file = save_test_results(args, test_result)
        print(f"🎉 [COMPLETE] Evaluation completed, accuracy: {test_result['accuracy']:.2f}%")
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "test_loss": test_result["loss"],
            "test_accuracy": test_result["accuracy"]
        })


if __name__ == '__main__':
    main() 