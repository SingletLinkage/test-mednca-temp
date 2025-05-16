import torch
import os
from tqdm import tqdm
from colorama import Fore, Back, Style
from src.agents.Agent import BaseAgent
from src.utils.helper import log_message

class Agent_GraphMedNCA(BaseAgent):
    def __init__(self, model, log_enabled=True, config=None):
        # The BaseAgent constructor only takes model as an argument
        super(Agent_GraphMedNCA, self).__init__(model)
        self.experiment = None  # Will be set later by Experiment class
        self.log_enabled = log_enabled
        self.projectConfig = config  # Store the config for later use
        
    def set_exp(self, experiment):
        """
        Set the experiment object for this agent
        This method is called by the Experiment class
        """
        self.experiment = experiment
        
        # Get logging preference from experiment config if available
        if experiment.get_from_config('verbose_logging') is not None:
            self.log_enabled = experiment.get_from_config('verbose_logging')
            
        # Initialize optimizer after experiment is set
        self.setup_optimizer()
        
    def setup_optimizer(self):
        """
        Initialize optimizer using experiment config
        Should be called after experiment is set
        """
        if self.experiment is None:
            return
            
        # Get optimizer parameters
        lr = self.experiment.get_from_config('lr')
        betas = self.experiment.get_from_config('betas')
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(lr),
            betas=betas
        )
        
        module = f"{__name__}" if self.log_enabled else ""
        log_message(f"Optimizer initialized with lr={lr}, betas={betas}", "SUCCESS", module, self.log_enabled, self.projectConfig)
    
    def process_epoch(self, epoch, loss):
        """
        Process each epoch - save model, log results
        """
        # Log to Tensorboard if available
        try:
            self.experiment.write_scalar("loss/train", loss, epoch)
        except:
            pass
            
        # Save model at intervals
        try:
            save_interval = self.experiment.get_from_config('save_interval')
            if epoch % save_interval == 0:
                save_path = os.path.join(self.experiment.get_from_config('model_path'), 
                                         'models', f'epoch_{epoch}')
                torch.save(self.model.state_dict(), save_path)
                
                module = f"{__name__}" if self.log_enabled else ""
                log_message(f"Model saved at epoch {epoch}", "SUCCESS", module, self.log_enabled, self.projectConfig)
        except Exception as e:
            module = f"{__name__}" if self.log_enabled else ""
            log_message(f"Error saving model: {str(e)}", "ERROR", module, self.log_enabled, self.projectConfig)
    
    def _get_actual_data(self, data_batch):
        """
        Get actual image data from filenames or whatever the dataset returns
        """
        # In this case, data_batch[0] contains filenames
        # We need to convert these to actual images using the Dataset's __getitem__
        if isinstance(data_batch, list) and len(data_batch) > 0:
            if isinstance(data_batch[0], tuple) and isinstance(data_batch[0][0], str):
                # These are filenames - we need the actual image data
                # Let's modify our approach to work with the Dataset_JPG class directly
                
                # We'll process this in the train method
                return None, None
            
        # If we reach here, try the other processing methods
        return self._process_data_batch(data_batch)
    
    def _process_data_batch(self, data_batch):
        """Helper method to extract images and labels from various data formats"""
        try:
            # Case 1: Dictionary format with 'img' and 'label' keys
            if isinstance(data_batch, dict):
                imgs = data_batch["img"]
                labels = data_batch["label"]
                
            # Case 2: Tuple/list of (img, label)
            elif isinstance(data_batch, (list, tuple)) and len(data_batch) == 2 and not isinstance(data_batch[0], tuple):
                imgs, labels = data_batch
                
            else:
                raise ValueError(f"Unsupported data format: {type(data_batch)}")
            
            # Move to device if tensors
            if hasattr(imgs, 'to'):
                imgs = imgs.to(self.model.device)
            if hasattr(labels, 'to'):
                labels = labels.to(self.model.device)
                
            return imgs, labels
            
        except Exception as e:
            module = f"{__name__}" if self.log_enabled else ""
            log_message(f"Error processing data batch: {str(e)}", "ERROR", module, self.log_enabled, self.projectConfig)
            return None, None
        
    def train(self, data_loader, loss_function):
        """
        Training loop adjusted for Graph-based Med-NCA
        """
        if self.experiment is None:
            raise ValueError("Experiment not set. Call set_exp() before training.")
            
        self.model.train()
        dataset = data_loader.dataset
        
        # Get configuration from experiment
        n_epoch = int(self.experiment.get_from_config('n_epoch'))
        steps = self.experiment.get_from_config('nca_steps')
        batch_size = self.experiment.get_from_config('batch_size')
        
        module = f"{__name__}" if self.log_enabled else ""
        log_message(f"Starting training for {n_epoch} epochs with {steps} NCA steps", "INFO", module, self.log_enabled, self.projectConfig)
        
        # Training loop
        for epoch in tqdm(range(n_epoch)):
            avg_loss = 0
            batch_count = 0
            total_images = len(dataset)
            
            log_message(f"Epoch {epoch+1}/{n_epoch}: Processing {total_images} images in batches of {batch_size}", "INFO", module, self.log_enabled, self.projectConfig)
            
            # Process images directly using dataset instead of dataloader
            # since the dataloader isn't providing the right format
            for batch_start in range(0, total_images, batch_size):
                batch_end = min(batch_start + batch_size, total_images)
                batch_indices = list(range(batch_start, batch_end))
                
                try:
                    # Get batch data directly from dataset
                    batch_data = []
                    batch_labels = []
                    
                    for idx in batch_indices:
                        item = dataset[idx]
                        if isinstance(item, tuple) and len(item) == 2:
                            img, label = item
                            batch_data.append(img)
                            batch_labels.append(label)
                    
                    if len(batch_data) == 0:
                        log_message(f"Warning: Empty batch from {batch_start} to {batch_end}", "WARNING", module, self.log_enabled, self.projectConfig)
                        continue
                        
                    # Stack tensors
                    imgs = torch.stack(batch_data).to(self.model.device)
                    labels = torch.stack(batch_labels).to(self.model.device)
                    
                    # Clear gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass with NCA steps from config
                    outputs = self.model(imgs, steps=steps)
                    
                    # Compute loss
                    loss = loss_function(outputs, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update average loss
                    avg_loss += loss.item()
                    batch_count += 1
                    
                    # Print occasional batch updates
                    if batch_count % 5 == 0:
                        log_message(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}", "STATUS", module, self.log_enabled, self.projectConfig)
                        
                except Exception as e:
                    log_message(f"Error processing batch {batch_start}-{batch_end}: {str(e)}", "ERROR", module, self.log_enabled, self.projectConfig)
                    import traceback
                    traceback.print_exc()
                    continue
                
            # Calculate average loss for the epoch (guard against division by zero)
            if batch_count > 0:
                avg_loss /= batch_count
                
            # Log progress
            log_message(f"Epoch {epoch+1}/{n_epoch}, Average Loss: {avg_loss:.4f}", "SUCCESS", module, self.log_enabled, self.projectConfig)
            
            # Process epoch (save checkpoints, etc.)
            self.process_epoch(epoch, avg_loss)

        save_path = os.path.join(self.experiment.get_from_config('model_path'), 'models', f'epoch_{epoch}')
        torch.save(self.model.state_dict(), save_path)
        log_message(f"Final Model saved at epoch {epoch}", "SUCCESS", module, self.log_enabled, self.projectConfig)
            
        log_message(" Training completed! ", "SUCCESS", module, self.log_enabled, self.projectConfig)

    def getAverageDiceScore_withimsave(self, output_dir=None):
        """
        Evaluate model on test set using Dice score and save segmentation maps
        
        This function does everything that getAverageDiceScore does plus saves the segmentation maps to files in runs/outputs directory
        """
        module = f"{__name__}" if self.log_enabled else ""
        
        if self.experiment is None:
            log_message("Experiment not set. Call set_exp() before evaluation.", "ERROR", module, self.log_enabled, self.projectConfig)
            return 0.0
            
        if output_dir is None:
            # Create output directory for segmentation maps
            output_dir = os.path.join(self.projectConfig[0]['model_path'], "outputs")
            os.makedirs(output_dir, exist_ok=True)
            log_message(f"Segmentation maps will be saved to {output_dir}", "INFO", module, self.log_enabled, self.projectConfig)
        else:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            log_message(f"Segmentation maps will be saved to {output_dir}", "INFO", module, self.log_enabled, self.projectConfig)
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Get the dataset and test indices
        try:
            dataset = self.experiment.dataset
            
            # Handle different implementations of DataSplit
            if hasattr(self.experiment.data_split, 'get_test_indices'):
                test_indices = self.experiment.data_split.get_test_indices()
            elif hasattr(self.experiment.data_split, 'test'):
                # If there's a 'test' attribute that contains indices
                test_indices = self.experiment.data_split.test
            else:
                # Fallback: try to access the test dictionary directly
                try:
                    # Assuming the data_split object has a dictionary structure with 'test' key
                    test_indices = list(range(len(dataset)))[-int(len(dataset) * 0.3):]  # Use last 30% as test by default
                    log_message("Using fallback test indices (last 30% of dataset)", "WARNING", module, self.log_enabled, self.projectConfig)
                except Exception as inner_e:
                    log_message(f"Could not determine test indices: {str(inner_e)}", "ERROR", module, self.log_enabled, self.projectConfig)
                    test_indices = []
            
            if not test_indices:
                log_message("No test data available for evaluation", "WARNING", module, self.log_enabled, self.projectConfig)
                return 0.0
                
            log_message(f"Evaluating model on {len(test_indices)} test images", "INFO", module, self.log_enabled, self.projectConfig)
            
            # Get inference steps from config
            steps = self.experiment.get_from_config('inference_steps')
            if steps is None:
                steps = 64  # Default to 64 steps if not specified in config
                log_message(f"No inference_steps in config, using default: {steps}", "WARNING", module, self.log_enabled, self.projectConfig)
            
            # For saving images
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # Calculate Dice score for each test image
            total_dice = 0.0
            with torch.no_grad():
                for idx in tqdm(test_indices):
                    # Get test image and label
                    item = dataset[idx]
                    
                    # Get image identifier for saving files
                    img_id = None
                    if isinstance(item, tuple) and len(item) == 3:
                        # Format: (image_id, image, label)
                        img_id, img, label = item
                    elif isinstance(item, tuple) and len(item) == 2:
                        # Format: (image, label)
                        img, label = item
                        img_id = f"image_{idx}"  # Create a default ID if none is provided
                    elif isinstance(item, dict):
                        # Format: {'image': img, 'label': label}
                        img = item['image']
                        label = item['label']
                        img_id = item.get('name', f"image_{idx}")  # Use 'name' key if available
                    else:
                        log_message(f"Unsupported item format: {type(item)}", "ERROR", module, self.log_enabled, self.projectConfig)
                        continue
                    
                    # Ensure proper tensor dimensions [batch, channel, height, width]
                    if len(img.shape) == 3:  # [channel, height, width]
                        img = img.unsqueeze(0)  # Add batch dimension
                    if len(label.shape) == 3:  # [channel, height, width]
                        label = label.unsqueeze(0)
                    
                    # Move to device
                    img = img.to(self.model.device)
                    label = label.to(self.model.device)
                    
                    # Get prediction WITH graph visualization data
                    prediction, graph_data = self.model(img, steps=steps, return_graph=True)
                    img_tensor, edge_index = graph_data
                    
                    # Apply sigmoid for binary segmentation (already done in forward)
                    
                    # Threshold predictions at 0.5 for binary segmentation
                    prediction_binary = (prediction > 0.5).float()
                    
                    # Compute Dice coefficient manually
                    # Flatten tensors
                    pred_flat = prediction_binary.view(-1)
                    label_flat = label.view(-1)
                    
                    # Calculate intersection and compute Dice
                    intersection = (pred_flat * label_flat).sum()
                    dice = (2. * intersection + 1) / (pred_flat.sum() + label_flat.sum() + 1)
                    
                    total_dice += dice.item()
                    
                    # Log dice score periodically
                    if idx % 10 == 0:
                        log_message(f"Test image {idx}: Dice score = {dice.item():.4f} (steps={steps})", "INFO", module, self.log_enabled, self.projectConfig)
                    
                    # Save segmentation maps and original images
                    try:
                        # Process filename to be safe for filesystems
                        if isinstance(img_id, str):
                            safe_filename = ''.join(c for c in img_id if c.isalnum() or c in '._-')
                        else:
                            safe_filename = f"image_{idx}"
                            
                        # Convert tensors to numpy for saving
                        orig_img = img[0].cpu().numpy()  # Remove batch dimension
                        pred_mask = prediction_binary[0].cpu().numpy()
                        true_mask = label[0].cpu().numpy()
                        
                        # Handle multi-channel images
                        if orig_img.shape[0] > 1:  # Multi-channel
                            if orig_img.shape[0] == 3:  # RGB
                                # Transpose from (C,H,W) to (H,W,C) for matplotlib
                                orig_img = np.transpose(orig_img, (1, 2, 0))
                            else:
                                # Just take first channel
                                orig_img = orig_img[0]
                        else:
                            orig_img = orig_img[0]
                            
                        # Prepare mask images (take first channel if multiple)
                        if len(pred_mask.shape) > 2 and pred_mask.shape[0] > 0:
                            pred_mask = pred_mask[0]
                        if len(true_mask.shape) > 2 and true_mask.shape[0] > 0:
                            true_mask = true_mask[0]
                        
                        # Create matplotlib figure with 2x2 grid instead of 1x3
                        fig = plt.figure(figsize=(15, 12))
                        gs = GridSpec(2, 2, figure=fig)
                        
                        # Original image - top left
                        ax1 = fig.add_subplot(gs[0, 0])
                        if len(orig_img.shape) == 3:  # Color image
                            ax1.imshow(orig_img)
                        else:  # Grayscale
                            ax1.imshow(orig_img, cmap='gray')
                        ax1.set_title("Original")
                        ax1.axis('off')
                        
                        # Prediction - top right
                        ax2 = fig.add_subplot(gs[0, 1])
                        ax2.imshow(pred_mask, cmap='gray')
                        ax2.set_title("Prediction")
                        ax2.axis('off')
                        
                        # Ground truth - bottom left
                        ax3 = fig.add_subplot(gs[1, 0])
                        ax3.imshow(true_mask, cmap='gray')
                        ax3.set_title("Ground Truth")
                        ax3.axis('off')
                        
                        # Graph visualization - bottom right
                        ax4 = fig.add_subplot(gs[1, 1])
                        if edge_index is not None:
                            from src.models.GraphMedNCA import visualize_graph
                            # Draw graph on the subplot
                            img_tensor_cpu = img_tensor.cpu()
                            # Create smaller figure for graph visualization
                            graph_fig = visualize_graph(img_tensor_cpu, edge_index)
                            # Convert graph_fig to image and display on ax4
                            ax4.imshow(self._fig2img(graph_fig))
                            plt.close(graph_fig)  # Close the temporary figure
                        else:
                            ax4.text(0.5, 0.5, "Graph visualization not available", 
                                    horizontalalignment='center', verticalalignment='center')
                        ax4.set_title("Graph Connections")
                        ax4.axis('off')
                        
                        # Add Dice score as text
                        plt.figtext(0.5, 0.01, f"Dice Score: {dice.item():.4f}", ha="center", fontsize=12,
                                  bbox={"facecolor":"orange", "alpha":0.8, "pad":5})
                        
                        # Save figure
                        plt.tight_layout()
                        save_path = os.path.join(output_dir, f"{safe_filename}_{dice.item():.2f}.png")
                        plt.savefig(save_path, bbox_inches='tight')
                        plt.close(fig)
                        
                    except Exception as save_error:
                        log_message(f"Error saving segmentation maps for image {img_id}: {str(save_error)}", "ERROR", module, self.log_enabled, self.projectConfig)
                        import traceback
                        traceback.print_exc()
            
            # Calculate average Dice score across all test images
            avg_dice = total_dice / len(test_indices) if test_indices else 0.0
            
            log_message(f"Evaluation completed. Average Dice Score: {avg_dice:.4f}", "SUCCESS", module, self.log_enabled, self.projectConfig)
            log_message(f"Segmentation maps saved to {output_dir}", "SUCCESS", module, self.log_enabled, self.projectConfig)
            return avg_dice
            
        except Exception as e:
            log_message(f"Error during evaluation: {str(e)}", "ERROR", module, self.log_enabled, self.projectConfig)
            import traceback
            traceback.print_exc()
            return 0.0

    def _fig2img(self, fig):
        """Convert a Matplotlib figure to a numpy array"""
        import numpy as np
        import io
        from PIL import Image
        
        # Save figure to a PNG in memory
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Load PNG into a numpy array via PIL
        img = np.array(Image.open(buf))
        
        # Close the buffer
        buf.close()
        
        return img
    
    def getAverageDiceScore(self):
        """
        Evaluate model on test set using Dice score
        """
        module = f"{__name__}" if self.log_enabled else ""
        
        if self.experiment is None:
            log_message("Experiment not set. Call set_exp() before evaluation.", "ERROR", module, self.log_enabled, self.projectConfig)
            return 0.0
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Get the dataset and test indices
        try:
            dataset = self.experiment.dataset
            
            # Handle different implementations of DataSplit
            if hasattr(self.experiment.data_split, 'get_test_indices'):
                test_indices = self.experiment.data_split.get_test_indices()
            elif hasattr(self.experiment.data_split, 'test'):
                # If there's a 'test' attribute that contains indices
                test_indices = self.experiment.data_split.test
            else:
                # Fallback: try to access the test dictionary directly
                try:
                    # Assuming the data_split object has a dictionary structure with 'test' key
                    test_indices = list(range(len(dataset)))[-int(len(dataset) * 0.3):]  # Use last 30% as test by default
                    log_message("Using fallback test indices (last 30% of dataset)", "WARNING", module, self.log_enabled, self.projectConfig)
                except Exception as inner_e:
                    log_message(f"Could not determine test indices: {str(inner_e)}", "ERROR", module, self.log_enabled, self.projectConfig)
                    test_indices = []
            
            if not test_indices:
                log_message("No test data available for evaluation", "WARNING", module, self.log_enabled, self.projectConfig)
                return 0.0
                
            log_message(f"Evaluating model on {len(test_indices)} test images", "INFO", module, self.log_enabled, self.projectConfig)
            
            # Get inference steps from config
            steps = self.experiment.get_from_config('inference_steps')
            # if steps is None:
            #     steps = 64  # Default to 64 steps if not specified in config
            #     log_message(f"No inference_steps in config, using default: {steps}", "WARNING", module, self.log_enabled, self.projectConfig)
            
            # Calculate Dice score for each test image
            total_dice = 0.0
            with torch.no_grad():
                for idx in tqdm(test_indices):
                    # Get test image and label
                    item = dataset[idx]
                    
                    if isinstance(item, tuple) and len(item) == 3:
                        # Format: (image_id, image, label)
                        _, img, label = item
                    elif isinstance(item, tuple) and len(item) == 2:
                        # Format: (image, label)
                        img, label = item
                    elif isinstance(item, dict):
                        # Format: {'image': img, 'label': label}
                        img = item['image']
                        label = item['label']
                    else:
                        log_message(f"Unsupported item format: {type(item)}", "ERROR", module, self.log_enabled, self.projectConfig)
                        continue
                    
                    # Ensure proper tensor dimensions [batch, channel, height, width]
                    if len(img.shape) == 3:  # [channel, height, width]
                        img = img.unsqueeze(0)  # Add batch dimension
                    if len(label.shape) == 3:  # [channel, height, width]
                        label = label.unsqueeze(0)
                    
                    # Move to device
                    img = img.to(self.model.device)
                    label = label.to(self.model.device)
                    
                    # Get prediction
                    prediction = self.model(img, steps=steps)
                    
                    # Apply sigmoid for binary segmentation
                    prediction = torch.sigmoid(prediction)
                    
                    # Threshold predictions at 0.5 for binary segmentation
                    prediction_binary = (prediction > 0.5).float()
                    
                    # Compute Dice coefficient manually
                    # Flatten tensors
                    pred_flat = prediction_binary.view(-1)
                    label_flat = label.view(-1)
                    
                    # Calculate intersection and compute Dice
                    intersection = (pred_flat * label_flat).sum()
                    dice = (2. * intersection + 1) / (pred_flat.sum() + label_flat.sum() + 1)
                    
                    total_dice += dice.item()
            
            # Calculate average Dice score across all test images
            avg_dice = total_dice / len(test_indices) if test_indices else 0.0
            
            log_message(f"Evaluation completed. Average Dice Score: {avg_dice:.4f}", "SUCCESS", module, self.log_enabled, self.projectConfig)
            return avg_dice
            
        except Exception as e:
            log_message(f"Error during evaluation: {str(e)}", "ERROR", module, self.log_enabled, self.projectConfig)
            import traceback
            traceback.print_exc()
            return 0.0
            
    def test(self, test_dir, output_dir, file_pattern=None):
        """
        Create segmentation maps for test images and save them to the output directory
        
        Args:
            test_dir (str): Directory containing test images
            output_dir (str): Directory to save segmentation maps
            file_pattern (str, optional): File pattern to match test images (e.g., '*.jpg')
                                         If None, process all image files
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image  # Still need PIL for loading images
        import torchvision.transforms as transforms
        
        module = f"{__name__}" if self.log_enabled else ""
        
        if self.experiment is None:
            log_message("Experiment not set. Call set_exp() before testing.", "ERROR", module, self.log_enabled, self.projectConfig)
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if test directory exists
        if not os.path.exists(test_dir):
            log_message(f"Test directory not found: {test_dir}", "ERROR", module, self.log_enabled, self.projectConfig)
            return
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Get inference steps from config
        steps = self.experiment.get_from_config('inference_steps')
        
        # Get transforms from dataset or create default
        if hasattr(self.experiment.dataset, 'transform'):
            transform = self.experiment.dataset.transform
        else:
            transform = transforms.ToTensor()
            
        # Get target size from config if available
        if self.experiment.get_from_config('input_size') is not None:
            target_size = tuple(self.experiment.get_from_config('input_size')[1])
        else:
            target_size = None
            
        # Get all image files in test directory
        if file_pattern:
            import glob
            image_files = glob.glob(os.path.join(test_dir, file_pattern))
        else:
            image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
            
        log_message(f"Found {len(image_files)} test images in {test_dir}", "INFO", module, self.log_enabled, self.projectConfig)
        
        # Process each test image
        with torch.no_grad():
            for image_file in tqdm(image_files, desc="Processing test images"):
                try:
                    # Extract filename without extension
                    filename = os.path.splitext(os.path.basename(image_file))[0]
                    
                    # Load image
                    img = Image.open(image_file)
                    
                    # Convert to RGB if not already
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    # Resize if target size specified
                    if target_size:
                        img = img.resize(target_size)
                        
                    # Apply transform
                    img_tensor = transform(img)
                    
                    # Add batch dimension if needed
                    if len(img_tensor.shape) == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                        
                    # Move to device
                    img_tensor = img_tensor.to(self.model.device)
                    
                    # Get prediction
                    prediction = self.model(img_tensor, steps=steps)
                    
                    # Apply sigmoid for binary segmentation
                    prediction = torch.sigmoid(prediction)
                    
                    # Threshold predictions at 0.5 for binary segmentation
                    prediction_binary = (prediction > 0.5).float()
                    
                    # Convert to numpy and remove batch dimension
                    mask = prediction_binary[0].cpu().numpy()
                    
                    # If mask has multiple channels, take the first one
                    if len(mask.shape) > 2 and mask.shape[0] == 1:
                        mask = mask[0]
                    
                    # Create figure and save using matplotlib
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(mask, cmap='gray')
                    ax.axis('off')
                    plt.tight_layout()
                    
                    # Save to output directory
                    output_path = os.path.join(output_dir, f"{filename}_mask.png")
                    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    
                    log_message(f"Saved mask for {filename} to {output_path}", "STATUS", module, self.log_enabled, self.projectConfig)
                    
                except Exception as e:
                    log_message(f"Error processing {image_file}: {str(e)}", "ERROR", module, self.log_enabled, self.projectConfig)
                    continue
                    
        log_message(f"Testing completed. Saved masks to {output_dir}", "SUCCESS", module, self.log_enabled, self.projectConfig)

