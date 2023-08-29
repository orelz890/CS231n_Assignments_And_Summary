import torch
import torch.nn as nn

import logging

# Configure logging to write to log files
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Get the logger for train
logger = logging.getLogger("train")
logger.addHandler(logging.FileHandler("train.log"))

# Get the logger for train
logger100 = logging.getLogger("train100")
logger100.addHandler(logging.FileHandler("train100.log"))

out_channels_1, out_channels_2, out_channels_3, num_classes = 16, 32, 64, 10
in_channels_1, in_channels_2, in_channels_3 = 3, 16, 32
filter_size_1, filter_size_2, filter_size_3 = 5, 3, 3


class NuralNetwork(nn.Module):
    def __init__(self):
        super(NuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels_1, out_channels_1, kernel_size=filter_size_1, padding=2),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels_2, out_channels_2, kernel_size=filter_size_2, padding=1),
            nn.BatchNorm2d(out_channels_2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels_3, out_channels_3, kernel_size=filter_size_3, padding=1),
            nn.BatchNorm2d(out_channels_3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4)),  # Add pooling layer
            nn.Flatten(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(out_channels_3 * 4 * 4, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = x.view(x.size(0), in_channels_1, 50, 50)  # Reshape the input tensor
        logits = self.linear_relu_stack(x)
        return logits
    

    def train_loop(self, dataloader, model, loss_fn, optimizer, device, epoch):
        logger.info(f"Epoche {epoch} \n---------------------\n")
        logger100.info(f"Epoche {epoch} \n---------------------\n")
        size = len(dataloader.dataset)
        correct = 0
        total = 0
        for batch, (X, y) in enumerate(dataloader):
            model.train()  # put model to training mode
            X = X.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            #Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred,y)

            #Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted_labels = torch.max(pred, 1)
            correct += (predicted_labels == y).sum().item()
            total += y.size(0)
            accuracy = 100 * correct / total

            loss, current = loss.item(), batch * len(X)
            logger.info(f"loss: {loss:>7f}\tAccuracy: {(accuracy):>7f}\tcurrect: [{correct:>7d}/{total:>5d}]\n")

            if batch %100 == 0:
                logger100.info(f"loss: {loss:>7f}\tAccuracy: {(accuracy):>7f}\tcurrect: [{correct:>7d}/{total:>5d}]\n")


    def test_loop(self, dataloader, model, loss_fn, device):
        logger.info(f"Test \n---------------------\n")
        logger100.info(f"Test \n---------------------\n")
        size = len(dataloader.dataset)
        test_loss, correct = 0, 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss/= size

        accuracy = 100 * correct / size

        logger.info(f"Test Error: \nAccuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}\n")
        logger100.info(f"Test Error: \nAccuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}\n")
        return accuracy
