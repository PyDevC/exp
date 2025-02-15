import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
# Define the hyperparameters
batch_size = 64 # The number of samples per batch
num_epochs = 10 # The number of times to iterate over the whole dataset
learning_rate = 0.01 # The learning rate for the optimizer

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.ToTensor(), # Convert the images to tensors
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize the pixel values with mean and std
])
# Load the MNIST dataset from the web
train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform) # The training set
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform) # The test set

# Create the data loaders for batching and shuffling the data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # The training loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # The test loader
# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The network has two fully connected layers
        self.fc1 = nn.Linear(28*28, 512) # The first layer takes the flattened image as input and outputs 512 features
        self.fc2 = nn.Linear(512, 10) # The second layer takes the 512 features as input and outputs 10 classes

    def forward(self, x):
        # The forward pass of the network
        x = x.view(-1, 28*28) # Flatten the image into a vector
        x = F.relu(self.fc1(x)) # Apply the ReLU activation function to the first layer
        x = self.fc2(x) # Apply the second layer
        return x # Return the output logits
# Create an instance of the model and move it to the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Get the device
model = Net().to(device) # Move the model to the device
print(model) # Print the model summary

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss() # The cross entropy loss for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # The stochastic gradient descent optimizer

# Define a function to calculate the accuracy of the model
def accuracy(outputs, labels):
    # The accuracy is the percentage of correct predictions
    _, preds = torch.max(outputs, 1) # Get the predicted classes from the output logits
    return torch.sum(preds == labels).item() / len(labels) # Return the ratio of correct predictions
# Define the training loop
def train(model, device, train_loader, criterion, optimizer, epoch):
    # Set the model to training mode
    model.train()
    # Initialize the running loss and accuracy
    running_loss = 0.0
    running_acc = 0.0
    # Loop over the batches of data
    for i, (inputs, labels) in enumerate(train_loader):
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs) # Get the output logits from the model
        loss = criterion(outputs, labels) # Calculate the loss
        # Backward pass and optimize
        loss.backward() # Compute the gradients
        optimizer.step() # Update the parameters
        # Print the statistics
        running_loss += loss.item() # Accumulate the loss
        running_acc += accuracy(outputs, labels) # Accumulate the accuracy
        if (i + 1) % 200 == 0: # Print every 200 batches
            print(f'Epoch {epoch}, Batch {i + 1}, Loss: {running_loss / 200:.4f}, Accuracy: {running_acc / 200:.4f}')
            running_loss = 0.0
            running_acc = 0.0

# Define the test loop
def test(model, device, test_loader, criterion):
    # Set the model to evaluation mode
    model.eval()
    # Initialize the loss and accuracy
    test_loss = 0.0
    test_acc = 0.0
    # Loop over the batches of data
    with torch.no_grad(): # No need to track the gradients
        for inputs, labels in test_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(inputs) # Get the output logits from the model
            loss = criterion(outputs, labels) # Calculate the loss
            # Print the statistics
            test_loss += loss.item() # Accumulate the loss
            test_acc += accuracy(outputs, labels) # Accumulate the accuracy
    # Print the average loss and accuracy
    print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc / len(test_loader):.4f}')
# Train and test the model for the specified number of epochs
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, criterion, optimizer, epoch) # Train the model
    test(model, device, test_loader, criterion) # Test the model

# Visualize some sample images and predictions
samples, labels = next(iter(test_loader)) # Get a batch of test data
samples = samples.to(device) # Move the samples to the device
outputs = model(samples) # Get the output logits from the model
_, preds = torch.max(outputs, 1) # Get the predicted classes from the output logits
samples = samples.cpu().numpy() # Move the samples back to CPU and convert to numpy array
fig, axes = plt.subplots(3, 3, figsize=(8, 8)) # Create a 3x3 grid of subplots
for i, ax in enumerate(axes.ravel()):
    ax.imshow(samples[i].squeeze(), cmap='gray') # Plot the image
    ax.set_title(f'Label: {labels[i]}, Prediction: {preds[i]}') # Set the title
    ax.axis('off') # Hide the axes
plt.tight_layout() # Adjust the spacing
plt.show() # Show the plot

