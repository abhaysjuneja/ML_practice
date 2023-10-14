import torch
import torch.nn as nn
import torch.optim as optim

from net import Net
from data import trainloader

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 2

for epoch in range(num_epochs):
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

PATH = "./cifar_net.pth"
torch.save(net.state_dict(), PATH)
