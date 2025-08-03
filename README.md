# -CNN-Fashion-MNIST-
练手作，除了模型源码外，还包含我自己训练后的模型参数，以及用于验证的十张图片与源码

请确保已安装了pytorch环境，防止代码无法正确运行

/dataset中为用于验证的数据集与Fashion-MNIST的原始数据集

FashionMNIST.pth为我训练后的模型参数，采用字典保存

model_Fashion_MNIST为模型训练的代码

test_Fashion-MNIST为验证十张图片的代码，运行后，如果已安装了tensorboard环境，可以在控制台输入"tensorboard --logdir=logs_FashionMNIST"后，点击网址进入网页查看载入模型的图片长什么样

因为只是入门没多久的萌新，还有许多不足。比如验证集选取的数目不够多，而且由于现在大部分图片像素比较高，压缩成28×28后很多图片特征都被压缩消失，比较容易识别错的为Sandal，Shirt（常常被认为成T-shirt或Pollover）。
如果有改进方法，欢迎告诉我，您的建议对我十分宝贵
