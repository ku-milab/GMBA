import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm
from typing import Tuple, Optional, List
from torch.autograd import Function
from collections import namedtuple
from utility import *






class CNNencoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU())

    def forward(self, x):
        out = self.model(x)
        return out



class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)



def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module



class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)




class ConditionalBatchNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channel)
        self.style = EqualLinear(512, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out




class Skip_CBN(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.model = spectral_norm(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False))
        self.cbn = ConditionalBatchNorm(channel)
        self.lrelu = nn.PReLU()

    def forward(self, x, skip_x, style, attr, mask):

        s1 = self.model(skip_x * mask * (1 + attr))
        s1 = self.cbn(s1, style)
        s1 = self.lrelu(s1)
        s2 = (s1 * mask) + (skip_x * (1 - mask))
        out = torch.cat((x, s2), 1)

        return out


class Bottle_CBN(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.model = spectral_norm(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False))
        self.cbn = ConditionalBatchNorm(channel)
        self.lrelu = nn.PReLU()

    def forward(self, x, style, attr, mask):

        b1 = self.model(x * mask * (1 + attr))
        b1 = self.cbn(b1, style)
        b1 = self.lrelu(b1)
        out = (b1 * mask) + (x * (1 - mask))

        return out





class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.E_conv1_1 = CNNencoder(3, 16)
        self.E_conv1_2 = CNNencoder(16, 16)
        self.E_conv2_1 = CNNencoder(16, 32)
        self.E_conv2_2 = CNNencoder(32, 32)
        self.E_conv3_1 = CNNencoder(32, 64)
        self.E_conv3_2 = CNNencoder(64, 64)
        self.E_conv4_1 = CNNencoder(64, 128)
        self.E_conv4_2 = CNNencoder(128, 128)
        self.E_conv5_1 = CNNencoder(128, 256)
        self.E_conv5_2 = CNNencoder(256, 256)
        self.E_conv6_1 = CNNencoder(256, 512)
        self.E_conv6_2 = CNNencoder(512, 256)

        self.G_conv1 = Bottle_CBN(256)
        self.G_conv1_2 = Skip_CBN(256)
        self.G_conv2 = CNNencoder(512, 128)
        self.G_conv2_2 = Skip_CBN(128)
        self.G_conv3 = CNNencoder(256, 64)
        self.G_conv3_2 = Skip_CBN(64)
        self.G_conv4 = CNNencoder(128, 32)
        self.G_conv4_2 = Skip_CBN(32)
        self.G_conv5 = CNNencoder(64, 16)
        self.G_conv5_2 = Skip_CBN(16)
        self.G_conv6 = CNNencoder(32, 8)
        self.G_out = nn.Sequential(
            spectral_norm(nn.Conv2d(8, 3, kernel_size=1, stride=1, bias=False)))


    def forward(self, x, style, attr, mask):
        a_2 = self.pooling(attr)  # (B, 128, 128)
        a_3 = self.pooling(a_2)  # (B, 64, 64)
        a_4 = self.pooling(a_3)  # (B, 32, 32)
        a_5 = self.pooling(a_4)  # (B, 16, 16)
        a_6 = self.pooling(a_5)  # (B, 8, 8)

        m_2 = self.pooling(mask)  # (B, 128, 128)
        m_3 = self.pooling(m_2)  # (B, 64, 64)
        m_4 = self.pooling(m_3)  # (B, 32, 32)
        m_5 = self.pooling(m_4)  # (B, 16, 16)
        m_6 = self.pooling(m_5)  # (B, 8, 8)

        c1 = self.E_conv1_1(x) # (B, 16, 256, 256)
        c1 = self.E_conv1_2(c1) # (B, 16, 256, 256)
        p1 = self.pooling(c1) # (B, 16, 128, 128)

        c2 = self.E_conv2_1(p1) # (B, 32, 128, 128)
        c2 = self.E_conv2_2(c2) # (B, 32, 128, 128)
        p2 = self.pooling(c2) # (B, 32, 64, 64)

        c3 = self.E_conv3_1(p2) # (B, 64, 64, 64)
        c3 = self.E_conv3_2(c3) # (B, 64, 64, 64)
        p3 = self.pooling(c3) # (B, 64, 32, 32)

        c4 = self.E_conv4_1(p3) # (B, 128, 32, 32)
        c4 = self.E_conv4_2(c4) # (B, 128, 32, 32)
        p4 = self.pooling(c4) # (B, 128, 16, 16)

        c5 = self.E_conv5_1(p4)  # (B, 256, 16, 16)
        c5 = self.E_conv5_2(c5)  # (B, 256, 16, 16)
        p5 = self.pooling(c5)  # (B, 256, 8, 8)

        c6 = self.E_conv6_1(p5) # (B, 512, 8, 8)
        c6 = self.E_conv6_2(c6) # (B, 512, 8, 8)

        u0 = self.G_conv1(c6, style, a_6, m_6)  # (B, 256, 8, 8)
        u1 = nn.Upsample(scale_factor=2).cuda()(u0)  # (B, 256, 16, 16)
        u1 = self.G_conv1_2(u1, c5, style, a_5, m_5)  # (B, (256+256), 16, 16)

        u1 = self.G_conv2(u1)  # (B, 128, 16, 16)
        u2 = nn.Upsample(scale_factor=2).cuda()(u1)  # (B, 128, 32, 32)
        u2 = self.G_conv2_2(u2, c4, style, a_4, m_4)  # (B, (128+128), 32, 32)

        u2 = self.G_conv3(u2)  # (B, 64, 32, 32)
        u3 = nn.Upsample(scale_factor=2).cuda()(u2)  # (B, 64, 64, 64)
        u3 = self.G_conv3_2(u3, c3, style, a_3, m_3)  # (B, (64+64), 64, 64)

        u3 = self.G_conv4(u3)  # (B, 32, 64, 64)
        u4 = nn.Upsample(scale_factor=2).cuda()(u3)  # (B, 32, 128, 128)
        u4 = self.G_conv4_2(u4, c2, style, a_2, m_2)  # (B, (32+32), 128, 128)

        u4 = self.G_conv5(u4)  # (B, 16, 128, 128)
        u5 = nn.Upsample(scale_factor=2).cuda()(u4)  # (B, 16, 256, 256)
        u5 = self.G_conv5_2(u5, c1, style, attr, mask)  # (B, (16+16), 256, 256)

        u5 = self.G_conv6(u5)  # (B, 8, 256, 256)
        out = self.G_out(u5)  # (B, 3, 256, 256)

        return out






class MappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        layers.append((EqualLinear(101, 512)))
        layers.append(nn.LeakyReLU(0.2))
        n_mlp = 7
        for i in range(n_mlp):
            layers.append(EqualLinear(512, 512))
            layers.append(nn.LeakyReLU(0.2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)

        return self.net(x)




class Conv_CBN_Dis(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()

        self.conv = spectral_norm(nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1, bias=False))
        self.cbn = ConditionalBatchNorm(out_c)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x, style):
        out = self.conv(x)
        out = self.cbn(out, style)
        out = self.lrelu(out)

        return out

class Conv_BN_Dis(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()

        self.conv = spectral_norm(nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1, bias=False))
        self.bn = nn.BatchNorm2d(out_c)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)

        return out



# PatchGAN Discriminator with Conditional BatchNorm (CBN)
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = Conv_CBN_Dis(3, 64, 4, 2)
        self.conv2 = Conv_BN_Dis(64, 128, 4, 2)
        self.conv3 = Conv_BN_Dis(128, 256, 4, 2)
        self.conv4 = Conv_BN_Dis(256, 512, 4, 2)
        self.out = nn.Sequential(spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)))


    def forward(self, x, style):

        x1 = self.conv1(x, style) # (B, 64, 109, 91)
        x2 = self.conv2(x1) # (B, 128, 54, 45)
        x3 = self.conv3(x2) # (B, 256, 27, 22)
        x4 = self.conv4(x3) # (B, 512, 13, 11)
        out = self.out(x4) # (B, 1, 12, 10)

        return out




class DEX_L2loss(nn.Module):

    def __init__(self, start_age, end_age):
        super().__init__()
        self.start_age = start_age
        self.end_age = end_age

    def forward(self, input, target):
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)
        # mean loss
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        loss = nn.MSELoss()(mean, target)

        return loss







"""
DexClassifier
"""
class DEXAgeClassifier(nn.Module):

    def __init__(self, vgg_path="./dex_imdb_wiki.caffemodel.pt", own_relu=False, outclass='all'):
        super().__init__()
        self.outclass = outclass
        self.vgg_path = vgg_path
        self.vgg_own_relu = own_relu
        self.classifier = VGG(own_relu=own_relu)
        vgg_state_dict = torch.load(vgg_path)
        vgg_state_dict = {k.replace('-', '_'): v for k, v in vgg_state_dict.items()}
        self.classifier.load_state_dict(vgg_state_dict)

    def __deepcopy__(self, memodict={}):
        return DEXAgeClassifier(self.vgg_path, own_relu=self.vgg_own_relu, outclass=self.outclass)

    def __call__(self, img, do_softmax=False):
        img = F.interpolate(img[:, [2, 1, 0]], size=(224, 224), mode='bilinear', align_corners=False) * 255
        age_pb = self.classifier(img)['fc8']
        age_pred = F.softmax(age_pb, 1) if do_softmax else age_pb
        return age_pred

    def get_hook(self):
        return self.classifier.conv5_3

    def get_classifier(self):
        return self.classifier

    @property
    def device(self):
        return next(self.classifier.parameters()).device

    def to(self, device):
        self.classifier = self.classifier.to(device)
        return self

    def zero_grad(self, *args, **kwargs):
        return self.classifier.zero_grad(*args, **kwargs)


class VGG(nn.Module):
    def __init__(self, pool='max', own_relu=False):
        super().__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.fc6 = nn.Linear(25088, 4096, bias=True)
        self.fc7 = nn.Linear(4096, 4096, bias=True)
        self.fc8_101 = nn.Linear(4096, 101, bias=True)

        self.own_relu = own_relu
        self.relu = nn.ReLU() if self.own_relu else F.relu

        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = {}
        out['r11'] = self.relu(self.conv1_1(x))
        out['r12'] = self.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = self.relu(self.conv2_1(out['p1']))
        out['r22'] = self.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = self.relu(self.conv3_1(out['p2']))
        out['r32'] = self.relu(self.conv3_2(out['r31']))
        out['r33'] = self.relu(self.conv3_3(out['r32']))
        out['p3'] = self.pool3(out['r33'])
        out['r41'] = self.relu(self.conv4_1(out['p3']))
        out['r42'] = self.relu(self.conv4_2(out['r41']))
        out['r43'] = self.relu(self.conv4_3(out['r42']))
        out['p4'] = self.pool4(out['r43'])
        out['r51'] = self.relu(self.conv5_1(out['p4']))
        out['r52'] = self.relu(self.conv5_2(out['r51']))
        out['r53'] = self.relu(self.conv5_3(out['r52']))
        out['p5'] = self.pool5(out['r53'])
        out['p5'] = out['p5'].view(out['p5'].size(0), -1)
        out['fc6'] = self.relu(self.fc6(out['p5']))
        out['fc7'] = self.relu(self.fc7(out['fc6']))
        out['fc8'] = self.fc8_101(out['fc7'])
        return out

    def __deepcopy__(self, memodict={}):
        m = VGG(own_relu=self.own_relu)
        m.load_state_dict(self.state_dict())
        return m





class GuidedBackpropReLUFunction(Function):
    @staticmethod
    def forward(self, input_img):
        output = F.relu(input_img).detach()
        self.save_for_backward((input_img > 0),)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img = self.saved_tensors[0]
        msk = input_img * (grad_output > 0)
        grad_input = grad_output * msk
        return grad_input

class GuidedBackpropReLU(torch.nn.Module):
    def forward(self,inp):
        return GuidedBackpropReLUFunction.apply(inp)


def module_no_grad(module):
    for x in module.parameters():
        x.requires_grad = False

def scale_prop_kernel(ref,tar,v):
    return int((tar/(ref/v))//2*2+1)

def getGaussianKernel(ksize,sigma=None):
    if sigma is None:
        sigma =  0.3*(ksize/2 - 1) + 0.8
    v = np.exp(-((np.arange(ksize)-(ksize-1)/2)**2)/(2*(sigma)**2))
    return (v / v.sum())[:,None]


def batch_blur(ims,ksize):
    k = torch.tensor(getGaussianKernel(ksize, None)).type_as(ims)
    b,c,w,h = ims.shape
    blur = torch.nn.functional.conv2d(ims.reshape(b*c,1,w,h),k[None,None,:],padding=((ksize-1)//2,0))
    blur = torch.nn.functional.conv2d(blur,k.T[None,None,:],padding=(0,(ksize-1)//2)).reshape(*ims.shape)
    return blur


class GradiendtClassifierWrapper(torch.nn.Module):
    def __init__(self, map_type='gb'):
        super().__init__()
        self.model = DEXAgeClassifier(vgg_path="./dex_imdb_wiki.caffemodel.pt", own_relu=False, outclass='all')

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU()

        recursive_relu_apply(self.model.get_classifier())
        module_no_grad(self.model.get_classifier())

        self.map_type = map_type

        self.gradients = None
        self.activations = None
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        [h.remove() for h in self.hooks]
        target_layer = self.model.get_hook()

        self.hooks = [
            target_layer.register_forward_hook(self.save_activation) ,
            target_layer.register_full_backward_hook(self.save_gradient) \
                if 'register_full_backward_hook' in dir(target_layer) \
                else target_layer.register_backward_hook(self.save_gradient)
        ]

    def save_activation(self, module, input, output):
        activation = output
        self.activations = activation.detach()

    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        self.gradients = grad.detach()

    def assert_device(self,x):
        if self.model.device != x.device:
            self.model.to(x.device)
            self.register_hooks()

    def forward(self, x):
        pred = self.model(x,do_softmax=False)
        return pred

    def __call__(self, x, **kwargs):
        if any(x.grad is not None for x in self.model.classifier.parameters()):
            self.model.zero_grad()

        self.gradients = None
        self.activations = None

        x.requires_grad = True
        x.grad = None
        h,w = x.shape[2:]

        self.assert_device(x)
        pred = super().__call__(x, **kwargs)

        if hasattr(self.model, 'topclass'):
            loss = self.model.topclass(pred).sum()
        else:
            loss = pred.sum()
        loss.backward()


        """
        Additional code for Grad-CAM
        """
        grad = self.gradients
        act = self.activations

        weights = torch.mean(grad, dim=(2, 3), keepdim=True)

        grad_cam = torch.sum(weights * act, dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-5)
        grad_cam = F.interpolate(grad_cam, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        grad_cam = (grad_cam / (grad_cam.std((1, 2, 3), keepdims=True) + 1e-15)).detach()

        grad_cam_scale = grad_cam / 10

        gb = x.grad.abs().mean(1, keepdims=True)
        gb = (gb / (gb.std((1, 2, 3), keepdims=True)+1e-15)).detach()

        blur_out = batch_blur((gb + grad_cam_scale),scale_prop_kernel(29, x.size(-1)/args.batch_size, 256))

        blur_out = blur_out.abs().sum(1, keepdims=True)
        thrs = blur_out.flatten(1).std(1).reshape(-1,1,1,1) * 2 + 1e-15
        blur_out = blur_out.clamp(max=thrs) / thrs

        out = blur_out.detach()

        threshold_ratio = 0.6
        out_sorted = torch.sort(out.view(-1)).values
        threshold_index = int((1 - threshold_ratio) * len(out_sorted))
        threshold_value = out_sorted[threshold_index]
        binary_mask = (out > threshold_value).float()

        atten = gb + grad_cam

        atten = (atten - atten.min()) / (atten.max() - atten.min())

        return atten, binary_mask




class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6)
        self.facenet.load_state_dict(torch.load("./model_ir_se50.pth"))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, weights):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        total_loss = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss = 1 - diff_target
            loss = weights[i] * loss

            total_loss += loss
        return total_loss / len(y)





class Backbone(Module):
	def __init__(self, input_size, num_layers, drop_ratio=0.4, affine=True):
		super(Backbone, self).__init__()
		assert input_size in [112, 224], "input_size should be 112 or 224"
		assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
		blocks = get_blocks(num_layers)

		unit_module = bottleneck_IR_SE
		self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
									  nn.BatchNorm2d(64),
									  nn.PReLU(64))
		if input_size == 112:
			self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
			                               nn.Dropout(drop_ratio),
			                               nn.Flatten(),
			                               nn.Linear(512 * 7 * 7, 512),
			                               nn.BatchNorm1d(512, affine=affine))
		else:
			self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
			                               nn.Dropout(drop_ratio),
			                               nn.Flatten(),
			                               nn.Linear(512 * 14 * 14, 512),
			                               nn.BatchNorm1d(512, affine=affine))

		modules = []
		for block in blocks:
			for bottleneck in block:
				modules.append(unit_module(bottleneck.in_channel,
										   bottleneck.depth,
										   bottleneck.stride))
		self.body = nn.Sequential(*modules)

	def forward(self, x):
		x = self.input_layer(x)
		x = self.body(x)
		x = self.output_layer(x)
		return l2_norm(x)

def l2_norm(input, axis=1):
	norm = torch.norm(input, 2, axis, True)
	output = torch.div(input, norm)
	return output


class bottleneck_IR_SE(Module):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR_SE, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = nn.MaxPool2d(1, stride)
		else:
			self.shortcut_layer = nn.Sequential(
				nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
				nn.BatchNorm2d(depth)
			)
		self.res_layer = nn.Sequential(
			nn.BatchNorm2d(in_channel),
			nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
			nn.PReLU(depth),
			nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
			nn.BatchNorm2d(depth),
			SEModule(depth, 16)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut



class SEModule(Module):
	def __init__(self, channels, reduction):
		super(SEModule, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x



class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
	""" A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
	return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
	if num_layers == 50:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=4),
			get_block(in_channel=128, depth=256, num_units=14),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 100:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=13),
			get_block(in_channel=128, depth=256, num_units=30),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 152:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=8),
			get_block(in_channel=128, depth=256, num_units=36),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	else:
		raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
	return blocks


def compute_cosine_weights(x):
	values = np.abs(x.cpu().detach().numpy())
	weights = 0.25 * (np.cos(np.pi * values)) + 0.75
	return weights




