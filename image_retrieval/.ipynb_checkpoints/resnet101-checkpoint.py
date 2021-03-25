import torch
import cv2
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import numpy as np
from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2] 按比例把宽W缩小到256
 transforms.CenterCrop(224),                #[3] 中间裁剪出224x224大小的图片
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        print('forward')
        for name, module in self.submodule._modules.items():
            #print(x.shape)
            #print(name)
            #print(module)
            if name == "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

extract_list = ["layer1", "avgpool", "fc"]

#加载参数
weights = torch.load('image_retrieval\\pretrained_model\\resnet101-5d3b4d8f.pth')
img = Image.open('cola.jpg')
#print(weights) 
#用参数加载模型
net = models.resnet101(pretrained=False)
net.load_state_dict(weights)
#print(net)
#预处理图片
'''
t = transforms.Resize(256)
img_t1 = t(img)
img_t1.show()
t2 = transforms.CenterCrop(224)
img_t2 = t2(img_t1)
img_t2.show()
'''
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
print(batch_t.shape)
#预测
net.eval()
out = net(batch_t)
print(out.shape)
_, index = torch.max(out, 1)

#标签
with open('image_retrieval\\labels.txt') as f:
  classes = [line.strip() for line in f.readlines()]
#
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(classes[index[0]], percentage[index[0]].item())
#
extract_result = FeatureExtractor(net, extract_list)

result = extract_result(batch_t)
feature = result[1].data.numpy()
print(feature.shape)
print(np.squeeze(feature).shape)
print(result[0].data.numpy().shape)