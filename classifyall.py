import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import median_filter
from sklearn.preprocessing import MinMaxScaler

# The CNN model
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

# Load the saved state dictionary into the model
device = torch.device('cpu')
num_classes = 21
model = CNN(num_classes).to(device)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Correct the orientation of the data
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

# Standardize the data
def standardise(data):
    dataset = np.zeros((data.shape[0], 1025))

    for index in range(len(data)):
        dataset[index][-1] = data[index][-1]
        row = data[index, :-1]
        row_indices_to_remove = []

        for i in range(len(row)):
            if row[i] < 0:
                row_indices_to_remove.append(i)
            elif row[i] > 255:
                row[i] /= 10
            if row[i] < 128:
                row[i] = 0
            else:
                row[i] = 255

        row = np.delete(row, row_indices_to_remove)
        dataset[index, :-1] = row

    return dataset

def preprocess_data(data):
    data = standardise(data)
    data = correct_orientation(data)
    return data

def main():
    test_data = pd.read_csv("testdata.txt", header=None).values

    n_datapoints = test_data.shape[0]
    test_data = preprocess_data(test_data)

    X_test = test_data[:, :1024].reshape(-1, 1, 32, 32).astype(np.float32)
    X_test = torch.tensor(X_test)

    with torch.no_grad():
        outputs = model(X_test)
        _, infer_labels = torch.max(outputs, 1)

    infer_labels = infer_labels.numpy()
    infer_labels = pd.DataFrame(infer_labels)

    assert type(infer_labels) == pd.DataFrame, f"infer_labels is of wrong type. It should be a DataFrame. type(infer_labels)={type(infer_labels)}"
    assert infer_labels.shape == (n_datapoints, 1), f"infer_labels.shape={infer_labels.shape} is of wrong shape. Should be {(n_datapoints, 1)}"
    
    infer_labels.to_csv("predlabels.txt", index=False, header=False)

if __name__ == "__main__":
    main()
