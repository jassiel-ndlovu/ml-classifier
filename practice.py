import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from scipy.ndimage import median_filter

# Apply median filter to reduce noise
def apply_median_filter(data, size=3):
    filtered_data = []
    for row in data:
        image = row[:1024].reshape(32, 32)
        filtered_image = median_filter(image, size=size)
        filtered_row = np.concatenate((filtered_image.flatten(), row[1024:]))
        filtered_data.append(filtered_row)
    return np.array(filtered_data)

# Correct the orientation
def correct_orientation(data):
    corrected_data = []
    for row in data:
        orientation = int(row[-1])
        image = row[:1024].reshape(32, 32)
        if orientation == 1:
            image = np.rot90(image, 3)  # 90 degrees clockwise
        elif orientation == 2:
            image = np.rot90(image, 2)  # 180 degrees
        elif orientation == 3:
            image = np.rot90(image, 1)  # 90 degrees counterclockwise
        corrected_row = np.concatenate((image.flatten(), row[1024:-1]))
        corrected_data.append(corrected_row)
    return np.array(corrected_data)

# This function removes all negative values in a row, divides all values greater than 255 by 10 and standardises values by changing
# a value to 0 if it is less than 128 and 255 otherwise.
def standardise(data):
    dataset = np.zeros((data.shape[0], 1025))

    for index in range(len(data)):
        dataset[index][-1] = data[index][-1]
        row = data[index, :-1]
        row_indices_to_remove = []

        for i in range(len(row)):
            # Check if a row is negative, and if so add it to the array
            if row[i] < 0:
                row_indices_to_remove.append(i)
            # As mentioned above, if the value is greater than 255 then divide by 10
            elif row[i] > 255:
                row[i] /= 10

            # Data standardisation: in our array we will now only have values 0 or 255
            if row[i] < 128:
                row[i] = 0
            else:
                row[i] = 255

        # We delete all the indices in the row that are negative
        row = np.delete(row, row_indices_to_remove)
        dataset[index, :-1] = row

    return dataset

def preprocess_data(data):
    # Standardise the data
    data = standardise(data)

    # Correct orientation and reduce noise
    data = correct_orientation(data)

    return data

# Define the CNN model using PyTorch
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = self.dropout3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)

    loss = running_loss / len(test_loader.dataset)
    accuracy = correct_predictions.double() / len(test_loader.dataset)
    return loss, accuracy

def display_images(data, labels, label_value, num_images=5):
    images = data[labels == label_value][:num_images]
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i, img in enumerate(images):
        img = img[:1024].reshape(32, 32)  # reshape to 32x32
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.show()

# Load data
trainDataFile = 'traindata.txt'
labelsFile = 'trainlabels.txt'
testDataFile = 'testdata.txt'
targetLabelsFile = 'targetlabels.txt'

# Read training data
with open(trainDataFile, 'r') as file:
    lines = file.readlines()
    matrix = [list(map(float, line.strip().split(','))) for line in lines]

# Read labels
with open(labelsFile, 'r') as file:
    lines = file.readlines()
    values = [float(line.strip()) for line in lines]

# Convert to numpy arrays
trainingData = np.array(matrix)
trainLabels = np.array(values)

# Preprocess data
X, y = preprocess_data(trainingData), trainLabels

# Convert to PyTorch tensors
X = torch.tensor(X[:, :1024].reshape(-1, 1, 32, 32), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 21
model = CNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Evaluate the model
loss, accuracy = evaluate_model(model, test_loader, criterion)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Read test data
with open(testDataFile, 'r') as file:
    lines = file.readlines()
    matrix = [list(map(float, line.strip().split(','))) for line in lines]

# Read target labels
with open(targetLabelsFile, 'r') as file:
    lines = file.readlines()
    values = [float(line.strip()) for line in lines]

# Convert to numpy arrays
testData = preprocess_data(np.array(matrix))
targetLabels = np.array(values)

# Convert to PyTorch tensors
X_test_additional = torch.tensor(testData[:, :1024].reshape(-1, 1, 32, 32), dtype=torch.float32)
y_test_additional = torch.tensor(targetLabels, dtype=torch.long)

# Create DataLoader for additional test data
test_additional_dataset = TensorDataset(X_test_additional, y_test_additional)
test_additional_loader = DataLoader(test_additional_dataset, batch_size=32, shuffle=False)

# Evaluate the model on additional test data
loss, accuracy = evaluate_model(model, test_additional_loader, criterion)
print(f"Additional Test Data Accuracy: {accuracy * 100:.2f}%")

# Save the model
torch.save(model.state_dict(), 'trained_model.pth')
