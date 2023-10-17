import argparse
from torch import nn, optim

# Define the model and loss function
model = ...
criterion = ...

def train(args):
    # Set up data loaders
    train_loader = ...
    val_loader = ...
    
    # Initialize the optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Train the model
    for epoch in range(args.epochs):
        for batch in train_loader:
            inputs, targets = batch
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validate the model
        with torch.no_grad():
            total_correct = 0
            for batch in val_loader:
                inputs, targets = batch
                
                # Forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                
                # Calculate accuracy
                correct = (predicted == targets).sum().item()
                total_correct += correct
            
            print('Epoch {}: Accuracy on validation set = {}'.format(epoch+1, total_correct / len(val_loader)))
        
        # Update learning rate
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    train(args)