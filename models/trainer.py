import torch 
from tqdm import tqdm

def train(dataloader, train_elems):
    # get train elements
    train_dataloader, eval_dataloader, _ = dataloader
    model = train_elems['model']
    optimizer = train_elems['optimizer']
    criterion = train_elems['criterion']
    num_epochs = train_elems['num_epochs']

    # train
    for epoch in range(num_epochs):
        model.train()
        train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in train_progress:
            optimizer.zero_grad()
            outputs = model(inputs['input_ids'].float())
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_progress.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(inputs))})
            
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for inputs, targets in eval_dataloader:
                outputs = model(inputs['input_ids'].float())
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%")


    