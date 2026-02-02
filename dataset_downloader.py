import os
from torchvision import datasets, transforms


    # 1. Define the path where you want the data
    # This points to a folder named 'data' inside your current project
data_folder_path = './data'

print(f"Checking for data in: {os.path.abspath(data_folder_path)}")

    # 2. Download Training Data
    # 'download=True' will create the folder if it doesn't exist
train_set = datasets.MNIST(
        root=data_folder_path, 
        train=True, 
        download=True,
        transform=transforms.ToTensor()
    )

    # 3. Download Test Data
test_set = datasets.MNIST(
        root=data_folder_path, 
        train=False, 
        download=True,
        transform=transforms.ToTensor()
    )

print("\nSUCCESS!")
print(f"Downloaded {len(train_set)} training images.")
print(f"Downloaded {len(test_set)} test images.")
print("Your 'data' folder is ready.")
