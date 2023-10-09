import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter   
import argparse
from statistics import mean
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1 , 8, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(8 , 16, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=256 , out_features=128)
        self.fc2 = nn.Linear(in_features=128 , out_features=64)
        self.fc3 = nn.Linear(in_features=64 , out_features=10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = torch.relu(self.conv1(x))       # First convolution followed by
        x = self.pool(x)                # a relu activation and a max pooling#
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = self.flatten(x)    

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return x


 # setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train(net, optimizer, loader, writer, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        writer.add_scalar('training loss', mean(running_loss), epoch)
        
        
def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total

import argparse

if __name__=='__main__':
    writer = SummaryWriter(f'runs/MNIST')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
    parser.add_argument('--epochs', type=int, default = 5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    args = parser.parse_args()
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
        
        # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # datasets
    trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
    testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    net = MNISTNet()
    # setting net on device(GPU if available, else CPU)
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)

    train(net, optimizer, trainloader, writer, epochs)
    test_acc = test(net, testloader)
    print(f'Test accuracy:{test_acc}')
  
    torch.save(net.state_dict(), 'weights/mnist_net.pth')
    
    #add embeddings to tensorboard
    perm = torch.randperm(len(trainset.data))
    images, labels = trainset.data[perm][:256], trainset.targets[perm][:256]
    images = images.unsqueeze(1).float().to(device)
    with torch.no_grad():
        embeddings = net.get_features(images)
        writer.add_embedding(embeddings,
                    metadata=labels,
                    label_img=images, global_step=1)

    # save networks computational graph in tensorboard
    writer.add_graph(net, images)
    # save a dataset sample in tensorboard
    img_grid = torchvision.utils.make_grid(images[:64])
    writer.add_image('mnist_images', img_grid)
    writer.close()