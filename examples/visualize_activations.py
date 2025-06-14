from what_the_kernel import ActivationVisualizer
import timm

def main():
    # List available models
    print("Available models in timm:")
    print(timm.list_models()[:5])  # Print first 5 models as example
    
    # Initialize visualizer with ResNet18
    visualizer = ActivationVisualizer(
        model_name="resnet18",
        layer_name="layer1.0.conv1"
    )
    
    # Print available layers
    print("\nAvailable layers in resnet18:")
    layers = visualizer.get_available_layers()
    print("\n".join(layers[:10]))  # Print first 10 layers as example
    
    # Load and visualize an image
    # Note: Replace with your image path
    image_path = "path/to/your/image.jpg"
    try:
        visualizer.load_image(image_path)
        visualizer.visualize(
            n_features=16,
            figsize=(15, 15),
            save_path="activation_visualization.png"
        )
    except FileNotFoundError:
        print(f"\nPlease provide a valid image path. Current path '{image_path}' not found.")

if __name__ == "__main__":
    main() 