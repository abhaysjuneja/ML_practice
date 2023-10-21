from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from net import Net

from data import trainloader

NUM_EPOCHS = 2
PATH = "./cifar_net.pth"


def train_setup() -> Dict[str, Any]:
    """setup classification network, loss function and optimizer based on whether gpu is used for training or not.

    Returns:
        Dict:
    """
    classifier = Net()
    loss_fn = nn.CrossEntropyLoss()
    sgd = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    return {"net": classifier, "criterion": loss_fn, "optimizer": sgd}


def train(
    net: Optional[nn.Module] = None,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
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
        for idx, data in enumerate(trainloader, 0):
            inputs, labels = data

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
    train_args = train_setup()
    train(**train_args)
