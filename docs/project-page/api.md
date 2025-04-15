# API Reference

## Model Architecture

The MNIST experiment uses a convolutional neural network defined in the `train.py` file:

```python
class Net(nn.Module):
    """CNN model for MNIST classification"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # Second convolutional layer
        self.dropout1 = nn.Dropout(0.25)  # Dropout layer
        self.dropout2 = nn.Dropout(0.5)   # Dropout layer
        self.fc1 = nn.Linear(9216, 128)   # First fully connected layer
        self.fc2 = nn.Linear(128, 10)     # Output layer (10 digits)

    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

## Training Functions

### Training Loop

```python
def train(args, model, device, train_loader, optimizer, epoch, results):
    """Train the model for one epoch"""
    model.train()
    # Training implementation details...
```

### Evaluation Function

```python
def test(model, device, test_loader, results, epoch=None):
    """Evaluate model on test data"""
    model.eval()
    # Testing implementation details...
```

## Data Preparation

The dataset is downloaded and prepared using the functions in `scripts/download_data.py`:

```python
def download_mnist_dataset(data_dir='./data'):
    """Download MNIST dataset to the specified directory"""
    # Implementation details...
```

## Command Line Options

The training script accepts the following command line options:

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size` | Input batch size for training | 64 |
| `--test-batch-size` | Input batch size for testing | 1000 |
| `--epochs` | Number of epochs to train | 5 |
| `--lr` | Learning rate | 1.0 |
| `--gamma` | Learning rate step gamma | 0.7 |
| `--no-cuda` | Disables CUDA training | False |
| `--no-mps` | Disables macOS GPU training | False |
| `--save-model` | Save the trained model | False | 