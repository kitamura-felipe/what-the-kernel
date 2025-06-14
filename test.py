from what_the_kernel import ActivationVisualizer
import timm
import os
import urllib.request
from PIL import Image
import torch


def get_layer_shapes(model):
    """Get shapes of all convolutional layers by running input through the network."""
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Dictionary to store layer shapes
    layer_shapes = {}
    
    # Hook function to capture shapes
    def hook_fn(name):
        def hook(module, input, output):
            layer_shapes[name] = output.shape
        return hook
    
    # Register hooks for all conv layers
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return layer_shapes


def get_conv_layers(model):
    """Get a list of all convolutional layer names."""
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]


def main():
    # Initialize visualizer with ResNet18
    print("\nInitializing ResNet18 visualizer...")
    model = timm.create_model("resnet18", pretrained=True)
    
    # Get shapes of all conv layers
    layer_shapes = get_layer_shapes(model)
    conv_layers = get_conv_layers(model)
    
    # Print available convolutional layers
    print("\nAvailable convolutional layers in resnet18:")
    print("\n{:<5} {:<20} {:<15} {:<15}".format("No.", "Layer Name", "Kernels", "Activation Size"))
    print("-" * 60)
    
    for i, name in enumerate(conv_layers, 1):
        shape = layer_shapes[name]
        kernels = shape[1]  # Number of output channels
        size = f"{shape[2]}x{shape[3]}"  # Height x Width
        print("{:<5} {:<20} {:<15} {:<15}".format(
            i, name, kernels, size
        ))
    
    # Get user input for layer selection
    while True:
        try:
            layer_num = int(input("\nEnter the number of the layer you want to visualize (1-{}): ".format(len(conv_layers))))
            if 1 <= layer_num <= len(conv_layers):
                break
            print(f"Please enter a number between 1 and {len(conv_layers)}")
        except ValueError:
            print("Please enter a valid number")
    
    selected_layer = conv_layers[layer_num - 1]
    print(f"\nSelected layer: {selected_layer}")
    
    # Initialize visualizer with the selected layer
    visualizer = ActivationVisualizer(
        model_name="resnet18",
        layer_name=selected_layer
    )
    
    # Download and load sample image
    image_path = "cat.jpg"
    
    try:
        print(f"\nLoading image from {image_path}")
        visualizer.load_image(image_path)
        
        print("Visualizing activations...")
        visualizer.visualize(
            n_features=256,
            figsize=(15, 15),
            cmap='gray',
            save_path="activation_visualization.png"
        )
        print("Visualization saved as 'activation_visualization.png'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 