import torch
import torchvision
import torchvision.transforms as transforms

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
    root="./.data", train=True, download=True, transform=transforms
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)


if __name__ == "__main__":
    print(len(trainloader))
