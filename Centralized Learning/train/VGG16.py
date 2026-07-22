from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

import src.utils.Log as Log

def train_on_device(model, learning, train_loader=None, device='cpu'):
    optimizer = optim.SGD(model.parameters(), lr=learning['learning-rate'], momentum=learning['momentum'])
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    data_iter = iter(train_loader)

    with tqdm(total=len(train_loader), desc="Processing", unit="step") as pbar:
        while True:
            # Training model
            model.train()
            optimizer.zero_grad()

            try:
                training_data, labels = next(data_iter)
                training_data = training_data.to(device)
                output = model(training_data)
                loss = criterion(output, labels)
                if torch.isnan(loss).any():
                    Log.print_with_color("NaN detected in loss", "yellow")
                loss.backward()

                optimizer.step()
                # tqdm bar
                pbar.update(1)

            except StopIteration:
                break

    Log.print_with_color("[>>>] Finish a round training!", "red")
    return model
