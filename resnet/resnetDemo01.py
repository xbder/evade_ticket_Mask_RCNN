from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

net = resnet152(pretrained=True)
print(net)