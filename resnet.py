import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, datasets
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset, random_split

import pandas as pd
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier

import random
import os
import math
import time
import copy

from PIL import Image

### device setup & miscellaneous
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} ..")

# Random seed: 42
random.seed(42)

# print(f"Vault Tensor: {vaults.shape}")
# print(vaults[:5])
# print(f"ICL Sizes: {icl_sizes.shape}")
# print(icl_sizes[:5])

### Data Prep
# Training    80
# validation  10
# testing     10

# resize images to 350 x 350
# normalization to each image using the color channels' mean and STD of all images in the training set
# image augmentation on training dataset:
# - horizontal flipping
# - random rotation (10% range)
# - random contrast (10% range)
# - random brightness (10% range)

### Custom Methods for UBM_Dataset
def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        img_tensor = TF.to_tensor(img)
        return img_tensor
    
# def is_valid_file(path):

#     return os.path.isfile(path) and path.endswith(".png")

# Custom ImageFolder class declaration to incorporate ICL sizes into images
class UBM_Dataset(datasets.DatasetFolder):
    def __init__(self, root, samples, transform=None, loader=None, is_valid_file=None, icl_size=None, labels=None):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.is_valid_file = is_valid_file
        self.icl_size = icl_size
        self.labels = labels

        self.samples = samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)
        
        target = self.labels[index]
        lense_size = self.icl_size[index]

        return image, target, lense_size
    
# Custom Layer class to incorporate icl_size
class ICL_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ICL_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.icl_size = None

        # Define the custom layer parameters
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.icl_param = nn.Parameter(torch.Tensor(1))

        self.weight.requires_grad = True
        self.bias.requires_grad = True
        self.icl_param.requires_grad = True

        # Initialize the custom layer parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.icl_param, -bound, bound)

    def set_icl_size(self, icl_size):
        self.icl_size = icl_size
    
    def forward(self, input):
        # Apply the custom layer operation using icl_size and input
        
        # print(f"adjusted_weight: {adjusted_weight.shape}")
        # print(f"adjusted_bias: {adjusted_bias.shape}")
        # print(f"self.weight: {self.weight.shape}")
        # print(f"self.bias: {self.bias.shape}")
        # print(f"self.icl_param: {self.icl_param.shape}")
        
        # Adjust the weight and bias based on icl_size
        adjusted_weight = self.weight * torch.exp(self.icl_param * self.icl_size)
        adjusted_bias = self.bias * torch.exp(self.icl_param * self.icl_size)

        # Perform the custom linear transformation
        output = torch.matmul(input, adjusted_weight) + adjusted_bias
        return output
        
# Custom Model Class to wrap the base model and added icl layer
class MyModel(nn.Module):
    def __init__(self, base_model, icl_layer):
        super(MyModel, self).__init__()
        self.base_model = base_model
        self.icl_layer = icl_layer
        self.icl_sizes = None
    
    def set_icl_sizes(self, icl_sizes):
        self.icl_sizes = icl_sizes
    
    def forward(self, inputs):
        images, icl_sizes = inputs
        # Pass the inputs through the base model
        base_outputs = self.base_model(images)
        # Pass the base model's output and icl_sizes to the custom layer
        # print(f"base_output: {base_outputs.shape}")

        icl_outputs = []
        batch_size = base_outputs.size(0)
        # print(batch_size)

        for i in range(batch_size):
            image_feature = base_outputs[i]                 # Current batch image
            self.icl_layer[1].set_icl_size(icl_sizes[i])    # set icl_size for self.icl_layer
            icl_output = self.icl_layer(image_feature)      # single output from image, icl_size pair
            icl_outputs.append(icl_output)                  # Concatenate outputs in the batch

        icl_outputs = torch.cat(icl_outputs, dim=0)

        return icl_outputs

# Data Transformation: resizing, contrast, and horizontal flipping (brightness, rotation)
data_transforms = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomAutocontrast(p=0.1)
])

# Create Data Loaders
def create_data_loaders(data_root, data_transforms, batch_size=8):
    ### Read excel file and extract vault sizes, ICL sizes -> tensor
    measures = pd.read_excel("./data/measures_ml.xlsx")
    vaults = measures["Post-Op Vault"]
    vaults = pd.to_numeric(vaults)
    vaults = torch.tensor(vaults).float()

    icl_sizes = measures["ICL Size"]
    icl_sizes = pd.to_numeric(icl_sizes)
    icl_sizes = torch.tensor(icl_sizes).float()

    # ubm images
    train_folders = []
    val_folders = []
    test_folders = []

    for i in range(311):
        # images (samples)
        train_samples = []
        val_samples = []
        test_samples = []
        # vault sizes (labels)
        train_labels = []
        val_labels = []
        test_labels = []
        # icl sizes (as additional variable)
        train_icl = []
        val_icl = []
        test_icl = []

        data_dir = os.path.join(data_root, f"Subject {i+1}")
        # print(data_dir)
        vault = vaults[i]
        icl_size = icl_sizes[i]
        # print(f"Vault: {vault}, icl_size: {icl_size}")

        image_names = os.listdir(data_dir)
        train_names, rest_names = train_test_split(image_names, test_size=0.2, random_state=42)
        val_names, test_names = train_test_split(rest_names, test_size=0.5, random_state=42)
        # print(f"train: {len(train_names)}, val: {len(val_names)}, test: {len(test_names)}")

        for name in train_names:
            if not name.startswith('.DS_Store'):
                train_samples.append(os.path.join(data_dir, name))
                train_labels.append(vault)
                train_icl.append(icl_size)
        for name in val_names: 
            if not name.startswith('.DS_Store'):
                val_samples.append(os.path.join(data_dir, name))
                val_labels.append(vault)
                val_icl.append(icl_size)
        for name in test_names:
            if not name.startswith('.DS_Store'):
                test_samples.append(os.path.join(data_dir, name))
                test_labels.append(vault)
                test_icl.append(icl_size)

        train_dataset = UBM_Dataset(root=data_dir, samples=train_samples, transform=data_transforms, loader=image_loader, labels=train_labels, icl_size=train_icl)
        val_dataset = UBM_Dataset(root=data_dir, samples=val_samples, transform=data_transforms, loader=image_loader, labels=val_labels, icl_size=val_icl)
        test_dataset = UBM_Dataset(root=data_dir, samples=test_samples, transform=data_transforms, loader=image_loader, labels=test_labels, icl_size=test_icl)

        train_folders.append(train_dataset)
        val_folders.append(val_dataset)
        test_folders.append(test_dataset)

    train_folder = torch.utils.data.ConcatDataset(train_folders)
    val_folder = torch.utils.data.ConcatDataset(val_folders)
    test_folder = torch.utils.data.ConcatDataset(test_folders)

    # print("Train dataset size:", len(train_folder))
    # print("Validation dataset size:", len(val_folder))
    # print("Test dataset size:", len(test_folder))

    # for i in range(10, 20):
    #     image, label, icl_size = test_folder[i]  # Get the image, label, and icl_size for the i-th item
    #     print(f"Item {i + 1}:")
    #     # print("  Shape:", image.shape)
    #     print("  Label:", label)
    #     print("  icl_size:", icl_size)

    train_loader = DataLoader(train_folder, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_folder, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_folder, batch_size=8, shuffle=True, num_workers=0)

    return train_loader, val_loader, test_loader

# Create Model given base model and icl layer
def make_model(base, icl_layer):
    model = MyModel(base, icl_layer)
    return model

# Hyperparameter tuning
def tune(model, loader, criterion, lr, patience, num_epochs=35):
    print("Starting the Cross-Validation ..")
    print()
    # Data Preparation
    num_folds = 10
    fold_size = len(loader.dataset) // num_folds
    # print(num_folds)
    # print(fold_size)
    indices = torch.randperm(len(loader.dataset))

    learning_rates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    for param in model.base_model.parameters():
        param.requires_grad = False

    for param in model.icl_layer[1].parameters():
        param.requires_grad = True

    # Get parameters for the custom layer only
    icl_params = filter(lambda p: p.requires_grad, model.icl_layer.parameters())
    icl_params = list(icl_params)  # Convert the filter object to a list

    optimizer = torch.optim.Adam(
        [{'params': icl_params, 'lr': 1}],
        lr=1
        )                               # Only activate the parameters in icl layer

    # Cross-Validation loop
    for fold_idx in range(num_folds):
        print(f"Fold # {fold_idx}")
        print()
        start = fold_idx * fold_size
        end = (fold_idx + 1) * fold_size if fold_idx < num_folds - 1 else len(loader.dataset)
        # print(start)
        # print(end)

        train_indices = []
        val_indices = indices[start:end]
        if start == 0:
            train_indices = indices[end:]
        elif end == len(loader.dataset):
            train_indices = indices[:start]
        else:
            train_indices = torch.cat((indices[:start], indices[end:]))

        train_dataset = Subset(loader.dataset, train_indices)
        val_dataset = Subset(loader.dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0)

        for l in learning_rates:
            print(f"Current learning rate: {l}")
            print()
            # training
            model.train()
            since = time.time()

            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = float('inf')
            no_improvement_c = 0

            # Update lr of Adam
            for g in optimizer.param_groups:
                g['lr'] = l

            for epoch in range(num_epochs):
                # print(f"Epoch {epoch}/{num_epochs-1}")
                # print("-" * 10)
                running_loss = 0.0

                for inputs, labels, icl_sizes in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    icl_sizes = icl_sizes.to(device)
                    model.set_icl_sizes(icl_sizes)
                
                    with torch.set_grad_enabled(True):
                        outputs = model((inputs, icl_sizes))
                        loss = criterion(outputs, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(train_loader.dataset)
                # print("Train Loss: {:4f}".format(epoch_loss))

                if epoch == 0 or epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    no_improvement_c = 0
                else:
                    no_improvement_c += 1
                
                if no_improvement_c >= patience:
                    print(f"No improvement in {patience} epochs. Stopping training early at epoch {epoch}.")
                    break
                    
                # print()
            
            time_elapsed = time.time() - since
            print("Training complete in {:0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            print("Best Train Loss: {:.4f}".format(best_loss))
            print()

            # Load the best model weights
            model.load_state_dict(best_model_wts)
            
            ### Evaluation
            since = time.time()
            model.eval()

            running_loss = 0.0

            for inputs, labels, icl_sizes in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                icl_sizes = icl_sizes.to(device)
                model.set_icl_sizes(icl_sizes)

                with torch.set_grad_enabled(True):
                    outputs = model((inputs, icl_sizes))
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(val_loader.dataset)
            print("Validation Loss: {:.4f}".format(epoch_loss))
            
            time_elapsed = time.time() - since
            print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            print()

# Hyperparameter tuning
def tune_res(loader, criterion, patience, num_epochs=35):
    print("Starting the Cross-Validation ..")
    print()
    # Data Preparation
    num_folds = 10
    fold_size = len(loader.dataset) // num_folds
    # print(num_folds)
    # print(fold_size)
    indices = torch.randperm(len(loader.dataset))

    models_res = [models.resnet18(), models.resnet34(), models.resnet50(), models.resnet101(), models.resnet152()]
    models_names = ["18", "34", "50", "101", "152"]

    # Cross-Validation loop
    for fold_idx in range(num_folds):
        print(f"Fold # {fold_idx}")
        print()
        start = fold_idx * fold_size
        end = (fold_idx + 1) * fold_size if fold_idx < num_folds - 1 else len(loader.dataset)
        # print(start)
        # print(end)

        train_indices = []
        val_indices = indices[start:end]
        if start == 0:
            train_indices = indices[end:]
        elif end == len(loader.dataset):
            train_indices = indices[:start]
        else:
            train_indices = torch.cat((indices[:start], indices[end:]))

        train_dataset = Subset(loader.dataset, train_indices)
        val_dataset = Subset(loader.dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0)

        for i in range(len(models_res)):
            print(f"Current model: ResNet {models_names[i]}")
            print()

            # take the last layer of given model and add additional layer via nn.Sequential
            model = models_res[i]
            lin = model.fc
            icl_layer = nn.Sequential(
                nn.ReLU(),
                ICL_Linear(lin.out_features, 1)
            )

            model = make_model(models_res[i], icl_layer)
            model.to(device)

            for param in model.base_model.parameters():
                param.requires_grad = False

            for param in model.icl_layer[1].parameters():
                param.requires_grad = True

            # Get parameters for the custom layer only
            icl_params = filter(lambda p: p.requires_grad, model.icl_layer.parameters())
            icl_params = list(icl_params)  # Convert the filter object to a list

            optimizer = torch.optim.Adam(
                [{'params': icl_params, 'lr': 0.00001}],
                lr=0.00001
            )
    
            # training
            model.train()
            since = time.time()

            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = float('inf')
            no_improvement_c = 0

            for epoch in range(num_epochs):
                # print(f"Epoch {epoch}/{num_epochs-1}")
                # print("-" * 10)
                running_loss = 0.0

                for inputs, labels, icl_sizes in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    icl_sizes = icl_sizes.to(device)
                    model.set_icl_sizes(icl_sizes)
                
                    with torch.set_grad_enabled(True):
                        outputs = model((inputs, icl_sizes))
                        loss = criterion(outputs, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(train_loader.dataset)
                # print("Train Loss: {:4f}".format(epoch_loss))

                if epoch == 0 or epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    no_improvement_c = 0
                else:
                    no_improvement_c += 1
                
                if no_improvement_c >= patience:
                    print(f"No improvement in {patience} epochs. Stopping training early at epoch {epoch}.")
                    break
                    
                # print()
            
            time_elapsed = time.time() - since
            print("Training complete in {:0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            print("Best Train Loss: {:.4f}".format(best_loss))
            print()

            # Load the best model weights
            model.load_state_dict(best_model_wts)
            
            ### Evaluation
            since = time.time()
            model.eval()

            running_loss = 0.0

            for inputs, labels, icl_sizes in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                icl_sizes = icl_sizes.to(device)
                model.set_icl_sizes(icl_sizes)

                with torch.set_grad_enabled(True):
                    outputs = model((inputs, icl_sizes))
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(val_loader.dataset)
            print("Validation Loss: {:.4f}".format(epoch_loss))
            
            time_elapsed = time.time() - since
            print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            print()

# Train Model
def train_model(model, loader, criterion, optimizer, patience, num_epochs=25, save_path=None):
    since = time.time()
    no_improvement_c = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf') # Initialize the best_loss with inf
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-" * 10)

        model.train()   # Set model to training mode

        running_loss = 0.0

        # Iterate over data
        for inputs, labels, icl_sizes in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            icl_sizes = icl_sizes.to(device)
            model.set_icl_sizes(icl_sizes)

            # print(f"inputs: {inputs.dtype}")
            # print(f"labels: {labels.dtype}")
            # print(f"icl_sizes: {icl_sizes.dtype}")

            # forward
            with torch.set_grad_enabled(True):
                # print(inputs.shape)
                outputs = model((inputs, icl_sizes))  # Pass both input images and icl sizes
                loss = criterion(outputs, labels)   # Calculate loss

                # backward + optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Stats
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(loader.dataset)
        print("Train Loss: {:4f}".format(epoch_loss))

        if epoch == 0 or epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improvement_c = 0
        else:
            no_improvement_c += 1

        if no_improvement_c >= patience:
            print(f"No improvement in {patience} epochs. Stopping training early at epoch {epoch}.")
            break

        print()
    
    time_elapsed = time.time() - since
    print("Training complete in {:0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best Train Loss: {:.4f}".format(best_loss))

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save best model weights to a file
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Best model weights saved to {save_path}")

    return model

def eval_model(model, loader, criterion, num_epochs=25):
    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-" * 10)

        model.eval()    # Set model to evaluate mode

        running_loss = 0.0

        for inputs, labels, icl_sizes in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            icl_sizes = icl_sizes.to(device)
            model.set_icl_sizes(icl_sizes)

            with torch.set_grad_enabled(True):
                outputs = model((inputs, icl_sizes))
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        print("Validation Loss: {:.4f}".format(epoch_loss))
    
    time_elapsed = time.time() - since
    print("Evaluation complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))


def main():

    ### get models
    # res18 = models.resnet18()
    # res34 = models.resnet34()
    # res50 = models.resnet50(pretrained=True)
    # res101 = models.resnet101(pretrained=True)
    res152 = models.resnet152()

    # take the last layer of given model and add additional layer via nn.Sequential
    lin = res152.fc
    icl_layer = nn.Sequential(
        nn.ReLU(),
        ICL_Linear(lin.out_features, 1)
    )

    model = make_model(res152, icl_layer)
    model.to(device)

    # From the Paper: 
    # Batch size: 8
    # Epochs: 25
    # Loss Function: mean squared error (MSE)
    # Optimizer: Adam's Optimizer
    # Batch size: 8
    # Initial Learning Rate: 10e-3
    # FREEZE THE RESNET18 PARAMETERS
    for param in model.base_model.parameters():
        param.requires_grad = False

    for param in model.icl_layer[1].parameters():
        param.requires_grad = True

    # Get parameters for the custom layer only
    icl_params = filter(lambda p: p.requires_grad, model.icl_layer.parameters())
    icl_params = list(icl_params)  # Convert the filter object to a list

    # Print icl_params to check if it contains any parameters
    # print("Parameters in the ICL layer:")
    # for param in icl_params:
    #     print("param")
    #     print(param.size())

    # Get images
    DATA_ROOT = "./data/ubm"

    EPOCHS = 35
    BATCH_SIZE = 4
    CRITERION = nn.L1Loss()
    LEARNING_RATE = 0.0001
    OPTIMIZER = torch.optim.Adam(
        [{'params': icl_params, 'lr': LEARNING_RATE}],
        lr=LEARNING_RATE
        )                               # Only activate the parameters in icl layer
    SAVE_PATH = "./model_weights/val/res18_basic.pth"
    PATIENCE = 5

    train_loader, val_loader, test_loader = create_data_loaders(DATA_ROOT, data_transforms, BATCH_SIZE)

    ### Hyperparameter tuning
    # tune(model, val_loader, CRITERION, LEARNING_RATE, PATIENCE, EPOCHS)
    # tune_res(val_loader, CRITERION, PATIENCE, EPOCHS)

    ### Training
    # trained = train_model(model, train_loader, CRITERION, OPTIMIZER, PATIENCE, EPOCHS, SAVE_PATH)
    
    ### Evaluation
    model.load_state_dict(torch.load(SAVE_PATH))
    eval_model(model, test_loader, CRITERION, 5)

if __name__ == "__main__":
    main()