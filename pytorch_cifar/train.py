from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from net import Net
from tqdm import tqdm

from data import trainloader

NUM_EPOCHS = 2
USE_MPS = False
USE_CUDA = True
PATH = "./cifar_net.pth"


def device_setup() -> Dict[str, Any]:
    """_summary_

    Returns:
        Dict[str, Any]: _description_
    """
    device = torch.device("cpu")

    if torch.backends.mps.is_available() and USE_MPS:
        device = torch.device("mps")
    elif torch.cuda.is_available() and USE_CUDA:
        device = torch.device("cuda:0")

    print(f"Using device {device}")
    return {"device": device}


def train_setup(device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    """setup classification network, loss function and optimizer based on whether gpu is used for training or not.

    Returns:
        Dict:
    """

    classifier = Net()
    classifier = classifier.to(device)

    loss_fn = nn.CrossEntropyLoss()
    sgd = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    return {"net": classifier, "criterion": loss_fn, "optimizer": sgd, "device": device}


def train(
    net: Optional[nn.Module] = None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
    **_,
) -> None:
    """training subroutine that takes in net, criterion and optimizer.

    Args:
        net (nn.Module): [description]
        criterion (nn.Module): [description]
        optimizer (optim.Optimizer): [description]
    """
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for idx, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            inputs, labels = data
            # move data batch to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = net(inputs)

            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss

            if idx % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Done")
    torch.save(net.state_dict(), PATH)


if __name__ == "__main__":
    device_args = device_setup()
    train_args = train_setup(**device_args)
    train(**train_args)
