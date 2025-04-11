import os
import argparse
from typing import Tuple
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

def download_mnist_dataset(data_dir: str = './data') -> Tuple[Dataset, Dataset]:
    """
    Download MNIST dataset to the specified directory
    
    Args:
        data_dir (str): Directory to store the dataset
        
    Returns:
        Tuple[Dataset, Dataset]: Training and test datasets
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download training data
    train_dataset = datasets.MNIST(
        data_dir, 
        train=True, 
        download=True,
        transform=transform
    )
    print(f"✅ Downloaded MNIST training dataset: {len(train_dataset)} samples")
    
    # Download test data
    test_dataset = datasets.MNIST(
        data_dir, 
        train=False, 
        download=True,
        transform=transform
    )
    print(f"✅ Downloaded MNIST test dataset: {len(test_dataset)} samples")
    
    return train_dataset, test_dataset

def main() -> None:
    parser = argparse.ArgumentParser(description='Download MNIST Dataset')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='directory to store the dataset (default: ./data)')
    args = parser.parse_args()
    
    print(f"📥 Downloading MNIST dataset to {args.data_dir}...")
    train_dataset, test_dataset = download_mnist_dataset(args.data_dir)
    print(f"✨ Done! Dataset saved to {os.path.abspath(args.data_dir)}")

if __name__ == '__main__':
    main()