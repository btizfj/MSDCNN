import os
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models


class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name

    def save(self, path):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)

    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")

    def load(self, path, modelfile=None):
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))

        if modelfile is None:
            model_files = glob.glob(complete_path + "/*")
            mf = max(model_files)
        else:
            mf = os.path.join(complete_path, modelfile)

        self.load_state_dict(torch.load(mf))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Bottleneck(nn.Module):

    def __init__(self, in_channels, mid_channels, dilation: int = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.ca = nn.Sequential(
            ChannelAttention(in_channels),
        )
        self.sa = nn.Sequential(
            SpatialAttention(3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + identity
        out = self.relu(out)
        return out


class SSDCNN(Model):

    def __init__(self, name, nclasses, cnn_name=None, class_names=None):
        super(SSDCNN, self).__init__(name)

        self.classnames = class_names
        self.nclasses = nclasses
        self.cnn_name = cnn_name
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.net = models.resnext50_32x4d(pretrained=True)
        self.net.fc = nn.Linear(2048, self.nclasses)

    def forward(self, x):
        x = self.net(x)
        return x


class MSDCNN(Model):

    def __init__(self, name, model, nclasses=None, pretraining=True, num_views=12, class_names=None):
        super(MSDCNN, self).__init__(name)
        self.classnames = class_names
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
        self.cnn = nn.Sequential(*list(model.net.children())[:-4])
        self.decoder = nn.Sequential(*list(model.net.children())[-4:-2])
        self.emb_dim = 512
        self.cls_dim = 2048
        self.classifier = nn.Linear(self.cls_dim, self.nclasses)
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ---------------------------------------------------#
        #   Encoder
        # ---------------------------------------------------#
        encoder_blocks = []
        for dilation in [2, 4, 6, 8]:
            encoder_blocks.append(
                Bottleneck(self.emb_dim, int(self.emb_dim / 2), dilation=dilation)
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)
        # init Encoder
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.transpose(0, 1)
        view_pool = []
        for v in x:
            v = self.cnn(v)
            view_pool.append(v)
        cat_views = view_pool[0]
        for i in range(1, len(view_pool)):
            cat_views = torch.max(cat_views, view_pool[i])
        encoder_feture = self.dilated_encoder_blocks(cat_views)
        decoder_feture = self.decoder(encoder_feture)
        out = self.AdaptiveAvgPool2d(decoder_feture)
        out = out.view(out.size(0), self.cls_dim * 1 * 1)
        out = self.classifier(out)
        return out
