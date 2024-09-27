import os
import json

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from torchvision import transforms

# from model import resnet50

import torch.nn as nn
import torch



class SimAM_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM_module, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambad=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return 'simam'

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)


# class Bottleneck_SimAM(nn.Module):
#     def __init__(self,c1,c2,shortcut=True,g=1,e=0.5):
#         super(Bottleneck_SimAM,self).__init__()
#         c_ = int(c2*e)
#         self.cv1 = Conv(c1,c_,1,1)
#         self.cv2 = Conv(c_,c2,3,1,g=g)
#         self.add = shortcut and c1 == c2
#         self.attention = SimAM_module(channels=c2)
#     def forward(self,x):
#         return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))



# class Channel_Att(nn.Module):
#     def __init__(self, channels=3, t=16):
#         super(Channel_Att, self).__init__()
#         self.channels = channels
#         self.bn2 = nn.BatchNorm2d(channels)

#     def forward(self, x):
#         residual = x
#         x = self.bn2(x)
#         weight_bn = torch.abs(self.bn2.weight) / torch.sum(torch.abs(self.bn2.weight))
#         x = x.permute(0, 2, 3, 1)
#         x = torch.mul(weight_bn, x)
#         x = x.permute(0, 3, 1, 2)
#         x = torch.sigmoid(x) * residual
#         return x

# class Att(nn.Module):
#     def __init__(self, channels=3, out_channels=None, no_spatial=True):
#         super(Att, self).__init__()
#         self.Channel_Att = Channel_Att(channels)

#     def forward(self, x):
#         x_out1 = self.Channel_Att(x)
#         return x_out1
# class Channel_Att(nn.Module):
#     def __init__(self, channels, t=16):
#         super(Channel_Att, self).__init__()
#         self.channels = channels

#         self.bn2 = nn.BatchNorm2d(self.channels, affine=True)


#     def forward(self, x):
#         residual = x

#         x = self.bn2(x)
#         weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = torch.mul(weight_bn, x)
#         x = x.permute(0, 3, 1, 2).contiguous()

#         x = torch.sigmoid(x) * residual #

#         return x


# class Att(nn.Module):
#     def __init__(self, channels, out_channels=None, no_spatial=True):
#         super(Att, self).__init__()
#         self.Channel_Att = Channel_Att(channels)

#     def forward(self, x):
#         x_out1=self.Channel_Att(x)

#         return x_out1
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.stride = stride

    #         self.nam = Att(out_channel)
    #         self.nam = Att(out_channel,no_spatial=self.no_spatial)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        #         out = self.nam(out)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, reduction=16,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        #         self.se = SELayer(out_channel * 4, reduction)
        self.attention = SimAM_module(channels=out_channel)

    #         self.nam = Att(out_channel*4)
    #         self.no_spatial = no_spatial
    #         self.nam = Att(out_channel * 4, no_spatial=self.no_spatial)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        #         out = self.se(out)
        out = self.attention(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        #         out = self.nam(out)
        out += identity
        out = self.relu(out)
        # attention_out = self.attention(out)  # 这里得到经过注意力机制后的输出
        # return attention_out, out  # 返回注意力后的特征及原始特征
        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 zero_init_residual=False,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        #          # 网络的第一层加入注意力机制
        #         self.ca = ChannelAttention(self.in_channel)
        #         self.sa = SpatialAttention()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        #           # 网络的卷积层的最后一层加入注意力机制
        #         self.ca1 = ChannelAttention(2048)
        #         self.sa1 = SpatialAttention()
        #         self.ca1 = Att(2048)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        #         for m in self.modules():
        #             if isinstance(m, nn.Conv2d):
        #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #         x = self.ca(x) * x
        #         x = self.sa(x) * x

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            #             x = self.ca1(x) * x
            #             x = self.sa1(x) * x

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = r"/home/ghz/ricetotal1/test/"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    # img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]
    img_path_list = []
    # for item in os.listdir(imgs_root):
    #     img_path_list.append(item)
    lable1 = []
    for i in os.listdir(imgs_root):
        ss = os.path.join(imgs_root, i)
        for j in os.listdir(ss):
            if j.endswith(".jpg"):
                img_path_list.append(j)
                lable1.append([ss, j])
    # read class_indict
    json_path = '/home/ghz/weights/class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet50(num_classes=5).to(device)

    # load model weights
    weights_path = "/home/ghz/trweights/resNet50_SAM3.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path),False)

    # prediction
    model.eval()
    batch_size = 5  # 每次预测时将多少张图片打包成一个batch
    preds = []
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in lable1[ids * batch_size: (ids + 1) * batch_size]:
                img_path = img_path[0] + "/" + img_path[1]
                # img_path = imgs_root + "/" + img_path
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)



            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)
            # preds, attention_weights = model(batch_img.to(device))
            # np_attention_weights = attention_weights[-1].squeeze().cpu().numpy()
            # # 可视化第一个图像的注意力权重
            # plt.imshow(np_attention_weights[0], cmap='hot', interpolation='nearest')
            # plt.show()
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))

                if class_indict[str(cla.numpy())] == 'bad':
                    preds.append(0)
                elif class_indict[str(cla.numpy())] == 'poor':
                    preds.append(1)
                elif class_indict[str(cla.numpy())] == 'good':
                    preds.append(2)
                elif class_indict[str(cla.numpy())] == 'better':
                    preds.append(3)
                elif class_indict[str(cla.numpy())] == 'medium':
                    preds.append(4)
        print(preds)
    print(len(preds))
    f = open("/home/ghz/weights/lables1.txt", "r")
    lines = f.readlines()  # 读取全部内容
    treds = []
    for line in lines:
        l = line.strip("\n").split(" ")
        treds.append(l)
        # print(line)
    #     print(treds)
    trues = []
    for item in treds:
        trues.append(int(item[1]))
    #     print(trues)
    acc = accuracy_score(trues, preds)
    acc_nums = accuracy_score(trues, preds, normalize=False)
    print(acc, acc_nums)  # 0.8 12

    # labels为指定标签类别，average为指定计算模式

    # micro-precision
    micro_p = precision_score(trues, preds, labels=[0, 1, 2], average='micro')
    # micro-recall
    micro_r = recall_score(trues, preds, labels=[0, 1, 2], average='micro')
    # micro f1-score
    micro_f1 = f1_score(trues, preds, labels=[0, 1, 2], average='micro')

    print(micro_p, micro_r, micro_f1)  # 0.8 0.8 0.8000000000000002

    # macro-precision
    macro_p = precision_score(trues, preds, labels=[0, 1, 2], average='macro')
    # macro-recall
    macro_r = recall_score(trues, preds, labels=[0, 1, 2], average='macro')
    # macro f1-score
    macro_f1 = f1_score(trues, preds, labels=[0, 1, 2], average='macro')

    print(macro_p, macro_r, macro_f1)

    # 计算混淆矩阵
    classes = list(set(trues))
    confusion_mtx = confusion_matrix(trues, preds, labels=classes)
    # 绘制混淆矩阵图
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_mtx, cmap=plt.cm.Blues)
    # 添加标签和标题
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    # 添加注释
    thresh = confusion_mtx.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, confusion_mtx[i, j],
                    ha='center', va='center',
                    color='white' if confusion_mtx[i, j] > thresh else 'black')
            # 显示图像
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()