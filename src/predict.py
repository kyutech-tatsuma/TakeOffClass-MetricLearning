import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from model import TakeoffClassModel

#Device Configuration
def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device(use_gpu=True)

# Define transformations to preprocess data
transform = transforms.Compose([transforms.ToTensor()])

#Load image data
img = Image.open('frame17.jpg')
inputs = transform(img)
inputs = inputs.unsqueeze(0)

#Model Loading
model = TakeoffClassModel()
model.load_state_dict(torch.load("model-test.pth"))

#Prediction Execution
with torch.no_grad():
    output = model(inputs)
    _, predicted = torch.max(output.data,1)

#Result Output
if predicted.item() == 0:
    print('result: 0')
else:
    print('result: 1')
