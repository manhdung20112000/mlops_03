import torch
import torchvision
import torchvision.transforms as transforms

def attempt_dataload(batch_size=4, seed=42, download=False):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    torch.manual_seed(seed)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, )

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader