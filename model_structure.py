import models
def get_model(netname, gpu, num_classes=100):
    """ return given network
    """
    if netname == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif netname == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_classes=num_classes)
    elif netname == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes=num_classes)
    elif netname == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif netname == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(num_classes=num_classes)
    elif netname == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif netname == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif netname == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif netname == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif netname == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif netname == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif netname == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif netname == 'xception':
        from models.xception import xception
        net = xception()
    elif netname == 'resnet18':
        from models.resnet import ResNet18
        net = ResNet18()
    elif netname == 'resnet34':
        from models.resnet import ResNet34
        net = ResNet34()
    elif netname == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=num_classes)
    elif netname == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=num_classes)
    elif netname == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif netname == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif netname == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif netname == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif netname == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif netname == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif netname == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif netname == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif netname == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif netname == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif netname == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif netname == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif netname == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif netname == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif netname == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif netname == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif netname == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif netname == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif netname == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif netname == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif netname == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif netname == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif netname == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif netname == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif netname == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif netname == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif netname == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')

    if gpu: #use_gpu
        net = net.cuda()

    return net