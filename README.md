# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## DESIGN STEPS

### STEP 1: 
Import required libraries and define image transforms.

### STEP 2: 
Load training and testing datasets using ImageFolder.


### STEP 3: 
Visualize sample images from the dataset.


### STEP 4: 
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.


### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.

## PROGRAM

### Name: GIFTSON RAJARATHINAM N
### Register Number: 212222233002

```python
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.models import VGG19_Weights
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

!unzip -qq ./chip_data.zip -d data

dataset_path = "./data/dataset"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

def show_sample_image(dataset, num_images = 5):
  fig, axes = plt.subplots(1, num_images, figsize=(5,5))
  for i in range(num_images):
    image, label = dataset[i]
    image = image.permute(1,2,0)
    axes[i].imshow(image)
    axes[i].set_title(dataset.classes[label])
    axes[i].axis('off')
  plt.show()

show_sample_image(train_dataset)

print(f"Total Number of training sample images: {len(train_dataset)}")

first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

print(f"Total Number of testing sample images: {len(test_dataset)}")

first_image, label = test_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model=models.vgg19(weights=VGG19_Weights.DEFAULT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
summary(model, input_size=(3, 224, 224))

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

summary(model, input_size=(3, 224, 224))

for param in model.features.parameters():
  param.requires_grad = False

criterion = nn.BCEWithLogitsLoss()
optimizer = opt.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, test_loader, num_epochs = 100):
  train_losses=[]
  val_losses=[]
  model.train()
  for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels.unsqueeze(1).float())
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
    train_losses.append(running_loss / len(train_loader))

    model.eval()
    val_loss=0.0
    with torch.no_grad():
      for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        val_loss+=loss.item()

    val_losses.append(val_loss / len(test_loader))
    model.train()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
  plt.figure(figsize=(10,5))
  plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker='o')
  plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker='s')
  plt.title("Training and Validation Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()
  print("NAME:GIFTSON RAJARATHINAM N")
  print("REGISTER.NO: 212222233002")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_model(model, train_loader, test_loader)

def test_model(model, test_loader):
  model.eval()
  correct = 0
  total = 0
  all_preds = []
  all_labels = []

  with torch.no_grad():
    for images, labels in test_loader:
      # Corrected the repeated assignment of images and labels
      # images, labels in test_loader
      images = images.to(device)
      labels = labels.float().unsqueeze(1).to(device)

      outputs = model(images)
      probs = torch.sigmoid(outputs)
      preds = (probs > 0.5).float()

      total += labels.size(0)
      correct += (preds == labels.int()).sum().item()

      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy().astype(int))

  accuracy = correct / total
  print(f"Test Accuracy: {accuracy:.4f}")


  class_names =['Negative', 'Positive']
  cm = confusion_matrix(all_labels, all_preds)
  print("NAME: GIFTSON RAJARATHINAM N")
  print("Register No: 212222233002")
  plt.figure(figsize=(8,6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()

  print("NAME: GIFTSON RAJARATHINAM N")
  print("Register No: 212222233002")
  print(classification_report(all_labels, all_preds, target_names=class_names))

# Define the predict_image function
def predict_image(model, image_index, dataset):
  model.eval()
  image, label = dataset[image_index]
  image = image.unsqueeze(0).to(device) # Add a batch dimension and move to device

  with torch.no_grad():
    output = model(image)
    prob = torch.sigmoid(output)
    pred = (prob > 0.5).float()

  predicted_class = dataset.classes[int(pred.item())]
  actual_class = dataset.classes[label]

  # Display the image and prediction
  plt.figure(figsize=(4,4))
  # Move the image back to CPU and permute dimensions for display
  plt.imshow(image.squeeze(0).cpu().permute(1, 2, 0))
  plt.title(f"Actual: {actual_class}, Predicted: {predicted_class}")
  plt.axis('off')
  plt.show()

test_model(model, test_loader)

predict_image(model, image_index=55, dataset=test_dataset)

predict_image(model,image_index=25, dataset=test_dataset)
```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2025-05-25 205155](https://github.com/user-attachments/assets/c5ed6b8c-d3d3-41ad-8ec7-e8f175b4b84e)


## Confusion Matrix

![Screenshot 2025-05-25 205222](https://github.com/user-attachments/assets/ce33c998-8f32-40fe-a9e8-f2b74a8d6715)


## Classification Report

![Screenshot 2025-05-25 205238](https://github.com/user-attachments/assets/7df20e59-1545-44f3-8e80-3c8c93305aaf)


### New Sample Data Prediction

![Screenshot 2025-05-25 205258](https://github.com/user-attachments/assets/b7302ad4-53f0-4a6f-9832-8ddf679cba60)


## RESULT
VGG19 model was fine-tuned and tested successfully. The model achieved good accuracy with correct predictions on sample test images.
