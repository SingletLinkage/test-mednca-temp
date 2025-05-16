import torch
import traceback
import os
import argparse
from colorama import Fore, Back, Style, init
from src.models.GraphMedNCA import GraphMedNCA
from src.datasets.Dataset_JPG import Dataset_JPG_Patch
from src.utils.Experiment import Experiment
from src.losses.LossFunctions import DiceBCELoss
from src.agents.Agent_GraphMedNCA import Agent_GraphMedNCA
from src.utils.helper import log_message, load_json_file, get_cur_run_id

# Initialize colorama
init(autoreset=True)

def main(model_path, log_enabled=True):
    try:
        module = f"{__name__}:main" if log_enabled else ""
        log_message("=== Initializing inference process ===", "INFO", module, log_enabled)
        
        # Check if model_path exists
        if not os.path.exists(model_path):
            log_message(f"Error: Model path does not exist: {model_path}", "ERROR", module, log_enabled)
            return
            
        # Load the config file
        config_path = os.path.join(model_path, 'config.dt')
        if not os.path.exists(config_path):
            log_message(f"Error: Config file not found at {config_path}", "ERROR", module, log_enabled)
            return
            
        # Load the configuration
        config = load_json_file(config_path)

        # Because theres some annoying thing that converts the input size to a list our code needs tuple
        if isinstance(config[0]['input_size'], list):
            config[0]['input_size'] = tuple(config[0]['input_size'])

        # log_message(f"DEBUG: INPUT SIZE {config[0]['input_size']}", "DEBUG", module, log_enabled)

        log_message(f"Loaded configuration from {config_path}", "SUCCESS", module, log_enabled, config)
        
        # Update model path in config to ensure it points to the correct location
        config[0]['model_path'] = model_path
        
        # Create output directory for inference results
        inference_dir = os.path.join(model_path, "inference_results")
        os.makedirs(inference_dir, exist_ok=True)
        log_message(f"Inference results will be saved to {inference_dir}", "INFO", module, log_enabled, config)
        
        # Set up logging for this inference run
        from datetime import datetime
        log_dir = os.path.join(model_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        config[0]['log_file'] = log_file
        config[0]['verbose_logging'] = log_enabled
        
        with open(log_file, 'w') as f:
            f.write("=== Inference Log ===\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Find the latest model checkpoint
        models_dir = os.path.join(model_path, 'models')
        if not os.path.exists(models_dir):
            log_message(f"Error: Models directory does not exist: {models_dir}", "ERROR", module, log_enabled, config)
            return
            
        # Find all epoch directories
        epoch_dirs = [d for d in os.listdir(models_dir) if d.startswith('epoch_')]
        if not epoch_dirs:
            log_message("No model checkpoints found in models directory", "ERROR", module, log_enabled, config)
            return
            
        # Sort by epoch number and find the latest checkpoint
        epoch_dirs.sort(key=lambda x: int(x.split('_')[1]))
        latest_epoch = epoch_dirs[-1]
        latest_model_path = os.path.join(models_dir, latest_epoch)
        
        # Check if the model file exists directly at this path (not inside a model.pth file)
        if not os.path.exists(latest_model_path):
            log_message(f"Error: Model checkpoint not found at {latest_model_path}", "ERROR", module, log_enabled, config)
            return
            
        log_message(f"Found latest model checkpoint: {latest_epoch}", "SUCCESS", module, log_enabled, config)
        
        # Initialize the model and dataset
        log_message("Setting up dataset for inference...", "INFO", module, log_enabled, config)
        dataset = Dataset_JPG_Patch(resize=True, log_enabled=log_enabled, config=config)
        
        # Define device
        device = torch.device(config[0].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        log_message(f"Using device: {device}", "INFO", module, log_enabled, config)
        
        # Initialize model with parameters from config
        hidden_channels = config[0].get('hidden_channels', 32)
        input_channels = config[0].get('input_channels', 1)
        fire_rate = config[0].get('fire_rate', 0.5)
        
        log_message("Initializing model...", "INFO", module, log_enabled, config)
        model = GraphMedNCA(
            hidden_channels=hidden_channels,
            n_channels=input_channels,
            fire_rate=fire_rate,
            device=device,
            log_enabled=log_enabled
        ).to(device)
        log_message("Model initialized successfully!", "SUCCESS", module, log_enabled, config)
        
        # Load model weights - directly from the path
        try:
            model.load_state_dict(torch.load(latest_model_path))
            log_message(f"Successfully loaded model weights from {latest_model_path}", "SUCCESS", module, log_enabled, config)
        except Exception as e:
            log_message(f"Error loading model weights: {str(e)}", "ERROR", module, log_enabled, config)
            traceback.print_exc()
            return
        
        # Initialize agent
        agent = Agent_GraphMedNCA(model, log_enabled=log_enabled, config=config)
        log_message("Agent initialized successfully!", "SUCCESS", module, log_enabled, config)
        
        # Initialize experiment
        exp = Experiment(config, dataset, model, agent, log_enabled=log_enabled)
        log_message("Experiment initialized successfully!", "SUCCESS", module, log_enabled, config)
        
        # Set model to evaluation mode
        exp.set_model_state('test')  # Use 'test' mode instead of 'train'
        dataset.set_experiment(exp)
        
        # Now we can check the dataset info
        log_message(f"Dataset size: {len(dataset)}", "INFO", module, log_enabled, config)
        
        # Skip inference if dataset is empty
        if len(dataset) == 0:
            log_message("Error: Dataset is empty. Check image and label paths.", "ERROR", module, log_enabled, config)
            return
        
        # Define loss function for metrics calculation
        loss_function = DiceBCELoss()
        
        # Run inference and save results
        if log_enabled:
            print(f"\n{Back.BLUE}{Fore.WHITE} RUNNING INFERENCE {Style.RESET_ALL}\n")
        else:
            print("\n=== RUNNING INFERENCE ===\n")
            
        # Run inference with image saving
        dice_score = agent.getAverageDiceScore_withimsave(output_dir=inference_dir)
        
        # Print summary
        log_message(f"Inference complete! Average Dice Score: {dice_score:.4f}", "SUCCESS", module, log_enabled, config)
        log_message(f"Results saved to: {os.path.join(model_path, 'outputs')}", "INFO", module, log_enabled, config)
        
    except Exception as e:
        log_message(f"Error in inference execution: {str(e)}", "ERROR", module, log_enabled, config if 'config' in locals() else None)
        traceback.print_exc()

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run inference for Graph-based MedNCA model')
    parser.add_argument('--model-path', type=str, 
                        default='/home/teaching/group21/final/test-mednca-temp/runs/model_0',
                        help='Path to the model directory')
    parser.add_argument('--log', action='store_true', help='Enable verbose logging with colors')
    parser.add_argument('--no-log', dest='log', action='store_false', help='Disable verbose logging')
    parser.set_defaults(log=True)
    
    args = parser.parse_args()
    main(model_path=args.model_path, log_enabled=args.log)
