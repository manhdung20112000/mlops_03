import json
import torch
from torch._C import device

from models.datasets import attempt_dataload
from models.model import Net

# load dataset
trainloader, testloader = attempt_dataload(batch_size=4, seed=42, download=False)

# model & device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'models/net.pth'
net = Net()
net.load_state_dict(torch.load(PATH, map_location=device))

# evaluate
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


metrics = {}
for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    metrics[classes[i]] = [acc]
    print('Accuracy of %5s : %2d %%' % (classes[i], acc))

# Serializing json 
json_object = json.dumps(metrics, indent = 4)
  
# Writing to sample.json
with open("metrics.json", "w") as outfile:
    outfile.write(json_object)