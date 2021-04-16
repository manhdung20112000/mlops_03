import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import json
from tqdm import tqdm

from models.model import Net
from models.datasets import attempt_dataload

# get dataloaders
batch_size = 4
seed = 42
trainloader, testloader = attempt_dataload(batch_size=batch_size, seed=42, download=False)

# device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    print(f"Training on {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")
except:
    print(f"Training on CPU")

# model
net = Net().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

progress_bar = tqdm(range(3))

for epoch in progress_bar:
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            progress_bar.set_description('epoch: %d | i: %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# save model
PATH = 'models/net.pth'
torch.save(net.state_dict(), PATH)