import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, test_loader, device, num_epochs, learning_rate, checkpoint_path):
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_test_loss = float("inf")
    best_train_accuracy = 0.0
    best_test_accuracy = 0.0
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Training loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        for batch in train_loader:
            # Expecting batch to be a tuple: (batch_data, batch_target)
            batch_data, batch_target = batch
            # Move each tensor in the dictionary to device
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            output = model(**batch_data)  # forward returns probabilities
            # We use log(output) since output is softmaxed
            loss = loss_fn(torch.log(output + 1e-8), batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = output.argmax(dim=1)
            train_correct += (preds == batch_target).sum().item()

        gc.collect()
        torch.cuda.empty_cache()

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)
        time_taken = time.time() - start_time

        # Validation loop
        model.eval()
        test_loss = 0.0
        test_correct = 0
        with torch.no_grad():
            for batch in test_loader:
                batch_data, batch_target = batch
                batch_data = {k: v.to(device) for k, v in batch_data.items()}
                batch_target = batch_target.to(device)
                output = model(**batch_data)
                loss = loss_fn(torch.log(output + 1e-8), batch_target)
                test_loss += loss.item()
                preds = output.argmax(dim=1)
                test_correct += (preds == batch_target).sum().item()
        
        gc.collect()
        torch.cuda.empty_cache()

        test_loss /= len(test_loader.dataset)
        test_accuracy = test_correct / len(test_loader.dataset)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | " +
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | Time: {time_taken:.4f}")
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_train_accuracy = train_accuracy
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print("New best checkpoint saved!")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model, best_train_accuracy, best_test_accuracy
