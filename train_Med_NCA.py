import torch
from src.datasets.Dataset_JPG_idk import Dataset_JPG
from src.models.Model_BackboneNCA import BackboneNCA
from src.losses.LossFunctions import DiceBCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_Med_NCA import Agent_Med_NCA
import os

config = [{
    'img_path': r"/home/teaching/group21/Dataset/2016_sample_img/2016_actual/2016_actual",
    'label_path': r"/home/teaching/group21/Dataset/2016_sample_img/2016_masked/2016_masked",
    'model_path': r'/home/teaching/group21/med_nca/M3D-NCA/model_path-noGraph',
    'device': "cuda:0" if torch.cuda.is_available() else "cpu",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    # Training
    'save_interval': 10,
    'evaluate_interval': 10,
    'n_epoch': 20,
    'batch_size': 1,
    # Model
    'channel_n': 16,
    'inference_steps': 64,
    'cell_fire_rate': 0.5,
    'input_channels': 16,  # Keep at 16
    'output_channels': 1,
    'hidden_size': 128,
    'train_model': 1,
    # Data dimensions - Minimum size should be at least 4x4 to allow for convolution operations
    'input_size': [(32, 32), (64, 64)],
    'data_split': [0.7, 0, 0.3],
    # Add minimum size parameter to enforce minimum dims
    'min_dimension': 8
}]

# Define Experiment
dataset = Dataset_JPG(resize=True)

img_path = config[0]['img_path']
label_path = config[0]['label_path']

# Get file lists
img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
label_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(img_files)} images in {img_path}")
print(f"Found {len(label_files)} labels in {label_path}")

# Match images with labels
matched_pairs = []
for img_file in img_files:
    img_base = os.path.splitext(img_file)[0]
    if img_file in label_files:
        matched_pairs.append((img_file, img_file))
    elif f"{img_base}_Segmentation.png" in label_files:
        matched_pairs.append((img_file, f"{img_base}_Segmentation.png"))
    elif f"{img_base}_segmentation.png" in label_files:
        matched_pairs.append((img_file, f"{img_base}_segmentation.png"))

print(f"Found {len(matched_pairs)} matching image-label pairs")

if matched_pairs:
    img_files_matched = [img for img, _ in matched_pairs]
    label_files_matched = [label for _, label in matched_pairs]
    dataset.setPaths(img_path, img_files_matched, label_path, label_files_matched)
else:
    print("ERROR: No matching image-label pairs found!")
    exit(1)

# Print image and label dimensions for debugging
print("Checking dataset item dimensions:")
if len(dataset) > 0:
    first_item = dataset[0]
    if isinstance(first_item, tuple) and len(first_item) >= 3:
        _, img, label = first_item
        print(f"Image dimensions before transformation: {img.shape}")
        print(f"Label dimensions: {label.shape}")
        
        # Ensure input_channels is 16
        config[0]['input_channels'] = 16
        print(f"Setting input_channels to: {config[0]['input_channels']}")

device = torch.device(config[0]['device'])

# Initialize models with input_channels=16
ca1 = BackboneNCA(
    config[0]['channel_n'], 
    config[0]['cell_fire_rate'], 
    device, 
    hidden_size=config[0]['hidden_size'],
    input_channels=config[0]['input_channels']  # Ensure this is used correctly in BackboneNCA
).to(device)

ca2 = BackboneNCA(
    config[0]['channel_n'], 
    config[0]['cell_fire_rate'], 
    device, 
    hidden_size=config[0]['hidden_size'],
    input_channels=config[0]['input_channels']  # Ensure this is used correctly in BackboneNCA
).to(device)

# Important: Set the input_channels attribute directly to ensure it's properly used
ca1.input_channels = config[0]['input_channels']
ca2.input_channels = config[0]['input_channels']

ca = [ca1, ca2]
agent = Agent_Med_NCA(ca)

# Make sure agent has correct input/output channels
agent.input_channels = config[0]['input_channels']
agent.output_channels = config[0]['output_channels']

exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')

# Adjust batch size
actual_batch_size = min(exp.get_from_config('batch_size'), len(dataset))
if actual_batch_size != exp.get_from_config('batch_size'):
    print(f"Warning: Batch size adjusted from {exp.get_from_config('batch_size')} to {actual_batch_size} due to dataset size")

# Custom collate function to transform input channels
def custom_collate(batch):
    ids = [item[0] for item in batch]
    images = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # Convert to tensor if not already
    images_tensor = torch.stack([img if isinstance(img, torch.Tensor) else torch.tensor(img) for img in images])
    labels_tensor = torch.stack([lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl) for lbl in labels])
    
    # Replicate single channel to 16 channels
    if images_tensor.shape[1] == 1:
        images_tensor = images_tensor.repeat(1, 16, 1, 1)
        print(f"Transformed batch from shape {images_tensor.shape[0], 1, images_tensor.shape[2], images_tensor.shape[3]} to {images_tensor.shape}")
    
    # Important: Ensure the dimensions are appropriate for the model
    h, w = images_tensor.shape[2], images_tensor.shape[3]
    # If dimensions are odd, pad to make them even for better downsampling
    if h % 2 != 0 or w % 2 != 0:
        pad_h = 0 if h % 2 == 0 else 1
        pad_w = 0 if w % 2 == 0 else 1
        images_tensor = torch.nn.functional.pad(images_tensor, (0, pad_w, 0, pad_h))
        labels_tensor = torch.nn.functional.pad(labels_tensor, (0, pad_w, 0, pad_h))
        print(f"Padded dimensions to: {images_tensor.shape}")
    
    return ids, images_tensor, labels_tensor

data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=actual_batch_size, collate_fn=custom_collate)

loss_function = DiceBCELoss()

# Print dimensions for debugging
print(f"Dataset size: {len(dataset)}")
print(f"Input size config: {config[0]['input_size']}")
print(f"Input channels: {config[0]['input_channels']}")
print(f"Device: {device}")
print(f"Model 1 input channels: {ca1.input_channels}")
print(f"Model 2 input channels: {ca2.input_channels}")

# For additional debugging, check model parameters
print("Model 1 parameters:")
for name, param in ca1.named_parameters():
    if 'fc0' in name:  # Focus on the linear layer causing the error
        print(f"{name}: {param.shape}")

# Wrap in a try/except to catch any errors
try:
    agent.train(data_loader, loss_function)
    print("Training completed successfully")
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()