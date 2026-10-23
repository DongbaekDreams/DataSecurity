from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm.auto import tqdm

'''Loss and fit functions taken from offical PyTorch documentation at:
https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html

Changes made to code are noted.
'''
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        
        # Added tqdm for progress bar.
        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

'''
Testing code taken from offical PyTorch documentation at:
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
Edits made to code are noted. Default comments are removed.
'''

# Added dataset_name to print the name of the dataset in the test loop.
def test_loop(dataloader, model, loss_fn, dataset_name):
    
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # for confusion matrix.
    actual = []
    predicted = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            actual.extend(y.tolist()) # for confusion matrix.
            predicted.extend(pred.argmax(1).tolist()) # for confusion matrix.


    test_loss /= num_batches
    correct /= size
    print(f"{dataset_name} Results:")
    print(f"Accuracy: {(100*correct):.1f}%")
    print(f"Average Loss: {test_loss:.4f}")
    
    ConfusionMatrixDisplay.from_predictions(actual, predicted, labels=range(10))
    plt.show()

'''
Data loading code taken from  offical PyTorch documentation at:
https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html;
'''
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32,
    scale=True)
    ])
)

# Renamed for consistency.
test_ds = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32,
        scale=True
        )])
)


# Added to create a validation set.
train_ds, validation_ds = random_split(
    training_data,
    [0.7, 0.3],
    generator=torch.Generator().manual_seed(67)
)


train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)
valid_dl = DataLoader(validation_ds, batch_size=64, shuffle=False) # Added


'''
Base model code taken from offical PyTorch documentation at:
https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
Changes made to code are noted.
'''
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Increased channels from 16 to 32, and reduced stride from 2 two 1.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        # Increased channels from 16 to 64; Increased input channels to match above.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Increased input channels to match above.
        self.conv3 = nn.Conv2d(64, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        # xb = F.relu(self.conv3(xb)) # Maybe explore removing this relu?
        xb = self.conv3(xb) # Removed after experimentation: Slightly improved accuracy.

        # Changed from (xb, 4) to (xb, 7) to allow for the changed input dimensions of the stride change.
        xb = F.avg_pool2d(xb, 7)
        return xb.view(-1, xb.size(1))





lr = 0.07 # Changed from 0.1 to 0.07 to improve accuracy.

epochs = 10

loss_func = F.cross_entropy 

torch.manual_seed(67)
model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl) #Valid to account for added validation set.

test_loop(valid_dl, model, loss_func, "Validation")
test_loop(test_dl, model, loss_func, "Test")
