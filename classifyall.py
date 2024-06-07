import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import median_filter
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = torch.load('trained_model.pth')
model.eval()  # Set the model to evaluation mode

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

# Apply median filter to reduce noise
def apply_median_filter(data, size=3):
    filtered_data = []
    for row in data:
        image = row[:1024].reshape(32, 32)
        filtered_image = median_filter(image, size=size)
        filtered_row = np.concatenate((filtered_image.flatten(), row[1024:]))
        filtered_data.append(filtered_row)
    return np.array(filtered_data)

# This function removes all negative values in a row, divides all values greater than 255 by 10 and standardises values by changing
# a value to 0 if it is less than 128 and 255 otherwise.
def standardise(data):
    for index in range(len(data)):
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
        data[index, :-1] = row

    return data

def preprocess_data(data):
    # First standardise the data
    data = standardise(data)

    # Correct all orientations
    data = correct_orientation(data)

    return data

def main():
    # Read test data
    test_data = pd.read_csv("testdata.txt", header=None).values

    # Preprocess test data
    test_data = preprocess_data(test_data)
    
    # Reshape the data for CNN
    X_test = test_data[:, :1024].reshape(-1, 1, 32, 32).astype(np.float32)
    X_test = torch.tensor(X_test)

    # Predict labels
    with torch.no_grad():
        outputs = model(X_test)
        _, infer_labels = torch.max(outputs, 1)

    infer_labels = infer_labels.numpy()
    infer_labels = pd.DataFrame(infer_labels)
    
    assert type(infer_labels) == pd.DataFrame, f"infer_labels is of wrong type. It should be a DataFrame. type(infer_labels)={type(infer_labels)}"
    assert infer_labels.shape == (test_data.shape[0], 1), f"infer_labels.shape={infer_labels.shape} is of wrong shape. Should be {(test_data.shape[0], 1)}"
    
    infer_labels.to_csv("predlabels.txt", index=False, header=False)

if __name__ == "__main__":
    main()
