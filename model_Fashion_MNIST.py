import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

#神经网络搭建
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    #数据集下载
    data_train = torchvision.datasets.FashionMNIST(root='./dataset',download=True,transform=transforms.Compose([transforms.ToTensor(),
                                                                                                                transforms.Normalize(0.2860,0.3530)]),train=True)
    data_test = torchvision.datasets.FashionMNIST(root='./dataset',download=True,transform=transforms.Compose([transforms.ToTensor(),
                                                                                                               transforms.Normalize(0.2860,0.3530)]),train=False)
    #数据集载入
    data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(data_test, batch_size=64, shuffle=True)

    #设置相关参数
    epoch = 10
    learning_rate = 0.001
    model = CNN().cuda()
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=5e-4)
    total_train_step = 0
    total_test_step = 0
    writer = SummaryWriter("logs_FashionMNIST")

    #开始训练
    for i in range(epoch):
        #训练部分
        print("-----第{}轮训练-----".format(i+1))
        model.train()
        for data in data_loader_train:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step % 100 == 0:
                print("第{}次训练，LOSS:{}".format(total_train_step, loss.item()))
                writer.add_scalar("loss_train", loss.item(), total_train_step)
        #测试部分
        model.eval()
        total_accuracy = 0
        total_loss = 0
        with torch.no_grad():
            for data in data_loader_test:
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                total_test_step += 1
                accuracy = (outputs.argmax(1) == labels).sum()
                total_accuracy += accuracy.item()
            print("测试数据集总损失：{}".format(total_loss))
            print("测试数据集的正确率为：{}".format(total_accuracy/len(data_test)))
            writer.add_scalar("loss_test", total_loss, total_test_step)
            writer.add_scalar("accuracy_test", total_accuracy/len(data_test), total_test_step)

    torch.save(model.state_dict(), "FashionMNIST.pth")
    writer.close()

