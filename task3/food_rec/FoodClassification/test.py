import torch
from torchvision import datasets, models, transforms
import os
import numpy
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np

input_size = 224

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class_dict = {0:'baklava',
              1:'pizza',
              2:'pomfri',
              3:'solata',
              4:'torta'}

images_path = '/home/miha/FRI/RIS/Exercise 9/FoodClassification/images'
model_path = '/home/miha/FRI/RIS/Foodero_dataset/Foodero/best_foodero_model.pt'

model = torch.load(model_path)
model.eval()

for image_name in os.listdir(images_path):
    image_file = os.path.join(images_path, image_name)

    img_p = Image.open(image_file)
    # img_p.show()
    # time.sleep(10)

    img = data_transforms['train'](img_p).unsqueeze(0)
    pred = model(img)

    pred_np = pred.cpu().detach().numpy().squeeze()
    class_ind = np.argmax(pred_np)

    print(pred_np)

    plt.imshow(img_p)
    plt.text(20,50,class_dict[class_ind], c='g')
    plt.show()

