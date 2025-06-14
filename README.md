# What The Kernel

A powerful tool for visualizing neural network activations in real-time using your webcam or YouTube videos. This project allows you to see how different layers of a neural network process visual information, making it an excellent educational and research tool for understanding deep learning models.

## Features

- Real-time visualization of neural network activations from your webcam
- Support for YouTube video processing
- Interactive layer selection for different neural network architectures
- Recording capability to save visualizations
- Support for all models available in the timm library
- Beautiful PyQt5-based GUI interface
- Real-time feature map visualization
- Easy-to-use API for custom implementations

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/what-the-kernel.git
cd what-the-kernel
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Webcam Visualization

To start the webcam visualization tool:

```bash
python -m what_the_kernel.camera_visualizer
```

This will open a GUI window with:
- Live webcam feed on the left
- Neural network activations on the right
- Layer selection dropdown
- Recording controls

### YouTube Video Visualization

To visualize activations from a YouTube video:

```python
from what_the_kernel.camera_visualizer import YouTubeVisualizerGUI
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
visualizer = YouTubeVisualizerGUI("YOUR_YOUTUBE_URL")
visualizer.show()
sys.exit(app.exec_())
```

### Custom Implementation

You can also use the core visualization functionality in your own code:

```python
from what_the_kernel import ActivationVisualizer

# Initialize the visualizer with a model from timm
visualizer = ActivationVisualizer(
    model_name="resnet18",  # Any model from timm
    layer_name="layer1.0.conv1"  # Layer to visualize
)

# Load and preprocess an image
image_path = "path/to/your/image.jpg"
visualizer.load_image(image_path)

# Visualize the activations
visualizer.visualize(
    n_features=16,
    figsize=(15, 15),
    save_path="activation_visualization.png"
)
```

## Available Models

This tool supports all models available in the timm library, including:
- ResNet variants (resnet18, resnet50, etc.)
- EfficientNet variants
- VGG variants
- MobileNet variants
- And many more!

To see all available models:
```python
import timm
print(timm.list_models())
```

## Requirements

- Python 3.7+
- PyQt5
- OpenCV
- PyTorch
- timm
- torchvision
- matplotlib
- numpy
- Pillow
- yt-dlp (for YouTube video support)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The timm library for providing access to various pre-trained models
- PyQt5 for the GUI framework
- OpenCV for video processing capabilities 