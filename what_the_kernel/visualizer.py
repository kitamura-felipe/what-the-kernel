import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, List, Tuple, Union

class ActivationVisualizer:
    def __init__(
        self,
        model_name: str = "vgg11",
        layer_name: str = "features.0",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the activation visualizer.
        
        Args:
            model_name (str): Name of the model from timm (default: 'vgg11')
            layer_name (str): Name of the layer to visualize (default: 'features.0')
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = model_name
        self.layer_name = layer_name
        
        # Load model
        self.model = timm.create_model(model_name, pretrained=True)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Get the target layer
        self.target_layer = self._get_layer(layer_name)
        if self.target_layer is None:
            raise ValueError(f"Layer {layer_name} not found in model {model_name}")
        
        # Register hook
        self.activations = None
        self._register_hook()
        
        # Initialize image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.input_image = None
        
    def _get_layer(self, layer_name: str) -> Optional[torch.nn.Module]:
        """Get the specified layer from the model."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def _hook_fn(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        """Hook function to store activations."""
        self.activations = output.detach()
    
    def _register_hook(self):
        """Register the hook to the target layer."""
        self.target_layer.register_forward_hook(self._hook_fn)
    
    def load_image(self, image_path: str):
        """
        Load and preprocess an image.
        
        Args:
            image_path (str): Path to the input image
        """
        image = Image.open(image_path).convert('RGB')
        self.input_image = self.transform(image).unsqueeze(0).to(self.device)
    
    def _normalize_activation(self, activation: torch.Tensor) -> np.ndarray:
        """Normalize activation values to [0, 1] range."""
        activation = activation.cpu().numpy()
        activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
        return activation
    
    def visualize(
        self,
        n_features: Optional[int] = None,
        figsize: Tuple[int, int] = (15, 15),
        save_path: Optional[str] = None,
        cmap: str = 'gray'
    ):
        """
        Visualize the activations.
        
        Args:
            n_features (int, optional): Number of feature maps to display. If None, shows half of total activations.
            figsize (tuple): Figure size for the plot
            save_path (str, optional): Path to save the visualization
            cmap (str): Colormap to use for visualization (default: 'gray')
        """
        if self.input_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Forward pass
        with torch.no_grad():
            self.model(self.input_image)
        
        if self.activations is None:
            raise ValueError("No activations captured. Check if the layer name is correct.")
        
        # Get activations and normalize
        activations = self._normalize_activation(self.activations[0])
        
        # If n_features is not specified, use half of total activations
        if n_features is None:
            n_features = activations.shape[0] // 2
        
        # Select features to display
        n_features = min(n_features, activations.shape[0])
        selected_features = np.arange(n_features)
        
        # Create visualization
        n_cols = int(np.ceil(np.sqrt(n_features)))
        n_rows = int(np.ceil(n_features / n_cols))
        
        plt.figure(figsize=figsize)
        for idx, feature_idx in enumerate(selected_features):
            plt.subplot(n_rows, n_cols, idx + 1)
            plt.imshow(activations[feature_idx], cmap=cmap)
            plt.title(f'Feature {feature_idx}')
            plt.axis('off')
        
        plt.suptitle(f'Activations from {self.layer_name} in {self.model_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()
    
    def get_available_layers(self) -> List[str]:
        """Get a list of all available layers in the model."""
        return [name for name, _ in self.model.named_modules()] 