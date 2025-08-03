import torch
import torchvision
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from model_Fashion_MNIST import CNN
from torch.utils.data import DataLoader, Dataset

#数据库类的创建
class MyDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('L')),
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize(0.2860, 0.3530),#因为训练时有归一化，验证时也需归一化才能正常识别
            transforms.Lambda(lambda img: 1-img),#灰度反转，使得照片为黑底白图
        ])
        self.images_list = os.listdir(self.root_dir)
    def __getitem__(self, index):
        image_name = self.images_list[index]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path)
        image = self.transform(image)
        return image, image_name
    def __len__(self):
        return len(self.images_list)
#dataset和dataloader构建
dataset = MyDataset('./dataset/test_FashionMNIST')
dataloader = DataLoader(dataset, batch_size=10)
#模型参数的加载
model = CNN()
model.load_state_dict(torch.load('FashionMNIST.pth'))
#字典建立，方便检验结果
class_dict=dict({0:"T-shirt",1:"Trouser",2:"Pullover",3:"Dress",4:"Coat",5:"Sandal",6:"Shirt",7:"Sneaker",8:"Bag",9:"Ankel-boot"})
#writer,方便检查图像
writer = SummaryWriter("logs_FashionMNIST")
step = 0
for image,image_name in dataset:
    writer.add_image("test",image,step)
    step += 1#写入图片，安装tensorboard后在控制台输入“tensorboard --logdir=logs_FashionMNIST”后，在出现的网址内查看载入的图片
#开始检验
model.eval()
with torch.no_grad():
    for images, name in dataloader:
        outputs = model(images)
        #print(outputs)
        predicted = outputs.argmax(1)
        for image,pred in zip(name, predicted):
            print("{}的预测结果为：{}".format(image,class_dict[pred.item()]))#输出各图片对应结果

writer.close()

