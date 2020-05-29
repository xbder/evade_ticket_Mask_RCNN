from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.resnet import BasicBlock, _resnet

# net = resnet18(pretrained=True)
# print(net)

# 也可以这样导入网络，第一个参数为网络名称
net18= _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=True, progress=True)
print(net18)
