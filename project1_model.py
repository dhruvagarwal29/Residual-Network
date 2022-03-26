
# importing all the important and reuqired libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from matplotlib import pyplot as plt


# for data augmentation - used RandomHorizontalFlip, RandomCrop, Normalize for train_data 
# and Normalize for test_data

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=[0, 2, 3, 4]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# setting batch size to be 64
batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# creating the basic block
class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        # adding drop out, here given p = 0.1 which gave the best results out of [0.1,0.2,0.3,0.4,0.5]
        self.dropout = nn.Dropout(p=0.1)
        
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.dropout(out) 
        out = F.relu(out)
        return out


#creating resnet model
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def project1_model():
    # using cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # creating the model
    net = ResNet(BasicBlock, [1,1,1,1]).to(device)

    # defining the loss and optimizer
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    #declaring our losses
    test_loss_history = []
    train_loss_history = []
    acc_history_train = []
    acc_history_test = []

    # training and testing for 200 epochs
    for epoch in range(200):

        train_loss,test_loss = 0.0,0.0
        correct_train,correct_test,total_train,total_test = 0,0,0,0

        for i, data in enumerate(trainloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predicted_output = net(images)
            fit = loss(predicted_output,labels)
            fit.backward()
            optimizer.step()
            train_loss += fit.item()
            _, predicted_output = torch.max(predicted_output.data, 1)
            correct_train += (predicted_output == labels).sum().item()
            total_train += labels.size(0)

        for i, data in enumerate(testloader):
            with torch.no_grad():
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                predicted_output = net(images)
                fit = loss(predicted_output,labels)
                test_loss += fit.item() 
                _, predicted_output = torch.max(predicted_output.data, 1)
                correct_test += (predicted_output == labels).sum().item()
                total_test += labels.size(0)

        train_loss = train_loss/len(trainloader)
        test_loss = test_loss/len(testloader)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        train_accuracy = (correct_train*100)/total_train
        test_accuracy = (correct_test*100)/total_test
        acc_history_train.append(train_accuracy)
        acc_history_test.append(test_accuracy)


        print('Epoch {0}, Train loss {1}, Train Accuracy {2}, Test loss {3}, Test Accuracy {4}'.format(
            epoch, train_loss, train_accuracy, test_loss, test_accuracy))

    # plotting the Test/Train accuracy vs Epoch
    def plot_graphs_accuracy(train_accuracy=[], test_accuracy=[]):
        plt.plot(acc_history_train)
        plt.plot(acc_history_test)
        max_acc_test = max(acc_history_test)
        max_accuracy_test_index = acc_history_test.index(max_acc_test)
        plt.plot(max_accuracy_test_index, max_acc_test, '.')
        plt.text(max_accuracy_test_index, max_acc_test, " Max Accuracy = {0}".format(
        max_acc_test))
        plt.legend(["train", "test"])
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.grid()
        plt.show()

    # plotting the Test/Train losses vs Epoch
    def plot_graphs_losses(train_loss_history=[], test_loss_history=[]):

        plt.plot(train_loss_history)
        plt.plot(test_loss_history)
        plt.legend(["train", "test"])
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title('Test/Train Loss Vs Epoch')
        plt.grid()
        plt.show()

    #calling both the above plotting functions    
    plot_graphs_accuracy(train_accuracy=acc_history_train,test_accuracy=acc_history_test)
    plot_graphs_losses(train_loss_history=train_loss_history, test_loss_history=test_loss_history)


    # Calculating the number parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('The Number of Parameters are: {0}'.format(count_parameters(net)))

    # saving the weights in .pt file
    model_path = './project1_model.pt'
    torch.save(net.state_dict(), model_path)
