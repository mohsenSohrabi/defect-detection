import torch 
from tqdm import tqdm

def train(dataloader, train_elems):
    # Unpack the dataloaders and training elements
    train_dataloader, eval_dataloader, _ = dataloader
    model = train_elems['model']
    optimizer = train_elems['optimizer']
    criterion = train_elems['criterion']
    num_epochs = train_elems['num_epochs']

    # Start the training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in train_progress:
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(inputs['input_ids'].float())  # Forward pass
            loss = criterion(outputs, targets)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights
            # Display the training loss
            train_progress.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(inputs))})
            
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to track gradients in validation phase
            total = 0
            correct = 0
            for inputs, targets in eval_dataloader:
                outputs = model(inputs['input_ids'].float())  # Forward pass
                _, predicted = torch.max(outputs.data, 1)  # Get the predicted classes
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            accuracy = 100 * correct / total  # Compute the accuracy
            # Print the validation accuracy for this epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%")
