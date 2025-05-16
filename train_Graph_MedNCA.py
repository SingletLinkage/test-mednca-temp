import torch
import torch.nn as nn
import traceback
import os
import shutil
import argparse
from colorama import Fore, Back, Style, init
from src.models.GraphMedNCA import GraphMedNCA
# Import the patched dataset from the new file
from src.datasets.Dataset_JPG import Dataset_JPG_Patch
from src.utils.Experiment import Experiment
from src.losses.LossFunctions import DiceBCELoss
from src.agents.Agent_GraphMedNCA import Agent_GraphMedNCA
from src.utils.helper import log_message

# Initialize colorama
init(autoreset=True)

def get_next_run_id(runs_dir):
    """
    Check if model_path and 'runs' subdirectory exist, create if needed.
    If it exists, determine the next run ID based on existing directories.
    """
    
    # Create directories if they don't exist
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
        return 0
    
    # Get existing run directories
    existing_runs = [d for d in os.listdir(runs_dir) if d.startswith('model_')]
    
    if not existing_runs:
        return 0
    
    # Extract run numbers and find the highest
    run_numbers = []
    for run_dir in existing_runs:
        try:
            run_num = int(run_dir.split('_')[1])
            run_numbers.append(run_num)
        except (IndexError, ValueError):
            pass
    
    return max(run_numbers) + 1 if run_numbers else 0


# Configure experiment
config = [{
    # 'img_path': r"/home/teaching/group21/Dataset/2016_sample_img/2016_actual/2016_actual",
    # 'label_path': r"/home/teaching/group21/Dataset/2016_sample_img/2016_masked/2016_masked",
    'img_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1-2_Training_Input",
    'label_path': r"/home/teaching/group21/Dataset/ISIC2018_Task1_Training_GroundTruth",
    # 'model_path': r'/home/teaching/group21/med_nca/M3D-NCA/model_path',
    'base_path': os.path.join(os.getcwd(), 'runs'),
    'device': "cuda",  # Use CPU for now for stability - change to cuda if needed
    'unlock_CPU': True,  # Add this to avoid thread limitation
    # Learning rate
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training config
    'save_interval': 5,
    'evaluate_interval': 2,
    'n_epoch': 25,   # Reduced to test the setup first
    'batch_size': 64,  # Reduced to avoid memory issues during debugging
    # Data
    'input_size': (64, 64),
    'data_split': [0.7, 0, 0.3], 
    # Graph-NCA parameters
    'hidden_channels': 32,  # Reduced for faster training during debugging
    'nca_steps': 16,
    'fire_rate': 0.5,
    # For JPG dataset
    'input_channels': 1,
    'output_channels': 1,
    # Logging configuration
    'verbose_logging': True,
    'inference_steps' : 32,
}]


def check_image_label_directories(img_path, label_path, log_enabled=True):
    """Check if the image and label directories exist and contain matching files"""
    module = f"{__name__}:check_image_label_directories" if log_enabled else ""
    
    # Check if directories exist
    if not os.path.exists(img_path):
        log_message(f"Error: Image directory does not exist: {img_path}", "ERROR", module, log_enabled, config=config)
        return False
        
    if not os.path.exists(label_path):
        log_message(f"Error: Label directory does not exist: {label_path}", "ERROR", module, log_enabled, config=config)
        return False
    
    # Get files from both directories
    img_files = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not img_files:
        log_message(f"Error: No image files found in {img_path}", "ERROR", module, log_enabled, config=config)
        return False
        
    if not label_files:
        log_message(f"Error: No label files found in {label_path}", "ERROR", module, log_enabled, config=config)
        # Print a sample of what files are there
        all_files = os.listdir(label_path)
        log_message(f"Files in label directory: {all_files[:10]}", "WARNING", module, log_enabled, config=config)
        return False
    
    # Check if there are matching files
    common_files = set(img_files).intersection(set(label_files))
    if not common_files:
        log_message(f"Warning: No matching filenames between image and label directories", "WARNING", module, log_enabled, config=config)
        log_message(f"Image files: {img_files[:5]}", "WARNING", module, log_enabled, config=config)
        log_message(f"Label files: {label_files[:5]}", "WARNING", module, log_enabled, config=config)
        # Note: We'll continue anyway since the Dataset_JPG_Patch class seems to handle this
    else:
        log_message(f"Found {len(common_files)} matching files between image and label directories", "SUCCESS", module, log_enabled, config=config)
    return True

def main(log_enabled=True):
    try:
        # set model path and run ID
        config[0]['run'] = get_next_run_id(config[0]['base_path'])
        config[0]['model_path'] = os.path.join(config[0]['base_path'], f"model_{config[0]['run']}")

        from datetime import datetime
        log_dir = os.path.join(config[0]['model_path'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d')}.log")
        config[0]['log_file'] = log_file

        with open(log_file, 'w') as f:
            f.write("=== Run Log ===\n")
            f.write(f"Run ID: {config[0]['run']}\n")
            f.write(f"Model Path: {config[0]['model_path']}\n")
            f.write(f"nca_steps: {config[0]['nca_steps']}\n")
            f.write(f"image_path: {config[0]['img_path']}\n")
            f.write(f"label_path: {config[0]['label_path']}\n")
            

        module = f"{__name__}:main" if log_enabled else ""
        log_message("=== Initializing training process ===", "INFO", module, log_enabled, config=config)
        
        # Update config with logging preference
        config[0]['verbose_logging'] = log_enabled

        log_message(f"Next run ID: {config[0]['run']}", "INFO", module, log_enabled, config=config)
        
        # Make sure the Dataset_JPG_Patch.py file exists and is importable
        patch_file = os.path.join(os.path.dirname(__file__), 'src', 'datasets', 'Dataset_JPG_Patch.py')
        if not os.path.exists(patch_file):
            log_message(f"Creating patched dataset file at {patch_file}...", "WARNING", module, log_enabled, config=config)
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(patch_file), exist_ok=True)
        
        # Check image and label directories before creating dataset
        img_path = config[0]['img_path']
        label_path = config[0]['label_path']
        
        log_message("Checking data directories...", "INFO", module, log_enabled, config=config)
        check_image_label_directories(img_path, label_path, log_enabled)
        
        # Use patched Dataset_JPG with resize=True
        log_message("Setting up dataset...", "INFO", module, log_enabled, config=config)
        dataset = Dataset_JPG_Patch(resize=True, log_enabled=log_enabled, config=config)
        device = torch.device(config[0]['device'])
        
        # Create the experiment first to set up the dataset
        log_message("Initializing model...", "INFO", module, log_enabled, config=config)
        nca = GraphMedNCA(
            hidden_channels=config[0]['hidden_channels'],
            n_channels=config[0]['input_channels'], 
            fire_rate=config[0]['fire_rate'],
            device=device,
            log_enabled=log_enabled
        ).to(device)
        log_message("GraphMedNCA Model initialized successfully!", "SUCCESS", module, log_enabled, config=config)
        
        agent = Agent_GraphMedNCA(nca, log_enabled=log_enabled, config=config)
        log_message("Agent_GraphMedNCA initialized successfully!", "SUCCESS", module, log_enabled, config=config)
        exp = Experiment(config, dataset, nca, agent, log_enabled=log_enabled)
        log_message("Experiment initialized successfully!", "SUCCESS", module, log_enabled, config=config)
        exp.set_model_state('train')
        dataset.set_experiment(exp)  # This should now initialize the dataset properly
        
        # Now we can check the dataset info
        log_message(f"Dataset size: {len(dataset)}", "INFO", module, log_enabled, config=config)
        
        # Skip training if dataset is empty
        if len(dataset) == 0:
            log_message("Error: Dataset is empty. Check image and label paths.", "ERROR", module, log_enabled, config=config)
            return
            
        # Test loading a sample from the dataset
        log_message("Testing dataset sample loading...", "INFO", module, log_enabled, config=config)
        try:
            sample_data, sample_label = dataset[0]
            log_message(f"Successfully loaded sample: data shape {sample_data.shape}, label shape {sample_label.shape}", "SUCCESS", module, log_enabled, config=config)
            
            # Test model forward pass on a single sample
            log_message("Testing model forward pass...", "INFO", module, log_enabled, config=config)
            with torch.no_grad():
                test_output = nca(sample_data.unsqueeze(0).to(device))
                log_message(f"Forward pass successful! Output shape: {test_output.shape}", "SUCCESS", module, log_enabled, config=config)
        except Exception as e:
            log_message(f"Error testing dataset: {e}", "ERROR", module, log_enabled, config=config)
            traceback.print_exc()
            return
            
        # Create data loader
        log_message("Creating data loader...", "INFO", module, log_enabled, config=config)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            shuffle=True, 
            batch_size=config[0]['batch_size'],
            num_workers=0  # Use 0 workers for debugging
        )
        
        loss_function = DiceBCELoss()
        
        # Print model parameters
        log_message(f"Model parameters: {sum(p.numel() for p in nca.parameters() if p.requires_grad)}", "INFO", module, log_enabled, config=config)
        
        # Train model
        if log_enabled:
            print(f"\n{Back.BLUE}{Fore.WHITE} STARTING TRAINING {Style.RESET_ALL}\n")
        else:
            print("\n=== STARTING TRAINING ===\n")
            
        agent.train(data_loader, loss_function)
        log_message("Training completed successfully!", "SUCCESS", module, log_enabled, config=config)
        
        # Evaluate model
        if log_enabled:
            print(f"\n{Back.BLUE}{Fore.WHITE} EVALUATING MODEL {Style.RESET_ALL}\n")
        else:
            print("\n=== EVALUATING MODEL ===\n")

        # agent.test(data_loader, loss_function)      
        dice_score = agent.getAverageDiceScore_withimsave()
        # agent.evaluate(data_loader, loss_function)


        if log_enabled:
            log_message(f"Evaluation complete! Average Dice Score: {dice_score}", "SUCCESS", module, log_enabled, config=config)
        else:
            print(f"Evaluation complete! Average Dice Score: {dice_score:.3f}")
        
    except Exception as e:
        log_message(f"Error in main execution: {str(e)}", "ERROR", module, log_enabled, config=config)
        traceback.print_exc()

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train Graph-based MedNCA model')
    parser.add_argument('--log', action='store_true', help='Enable verbose logging with colors')
    parser.add_argument('--no-log', dest='log', action='store_false', help='Disable verbose logging')
    parser.set_defaults(log=True)
    
    args = parser.parse_args()
    main(log_enabled=args.log)