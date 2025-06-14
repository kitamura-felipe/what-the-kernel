import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from .visualizer import ActivationVisualizer
import sys

class CameraVisualizerGUI(QMainWindow):
    def __init__(self, model_name="resnet18"):
        super().__init__()
        self.setWindowTitle("Neural Network Activation Visualizer")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
            
        # Get camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize recording variables
        self.is_recording = False
        self.video_writer = None
        self.recorded_frames = []
        
        # Initialize model and get available layers
        self.model_name = model_name
        self.visualizer = ActivationVisualizer(model_name, "layer1.0.conv1")
        self.available_layers = self.visualizer.get_available_layers()
        self.conv_layers = [name for name in self.available_layers 
                          if any(name.endswith(f"conv{i}") for i in range(1, 10))]
        
        # Create GUI elements
        self.setup_gui()
        
        # Initialize transform for the model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Setup timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)  # Update every 10ms
        
    def setup_gui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create horizontal layout for video and activation displays
        display_layout = QHBoxLayout()
        
        # Create video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        display_layout.addWidget(self.video_label)
        
        # Create activation display
        self.activation_label = QLabel()
        self.activation_label.setMinimumSize(640, 480)
        display_layout.addWidget(self.activation_label)
        
        layout.addLayout(display_layout)
        
        # Create controls layout
        controls_layout = QHBoxLayout()
        
        # Create layer selection dropdown
        self.layer_combo = QComboBox()
        self.layer_combo.addItems(self.conv_layers)
        self.layer_combo.currentTextChanged.connect(self.on_layer_change)
        controls_layout.addWidget(self.layer_combo)
        
        # Create record button
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.record_button)
        
        layout.addLayout(controls_layout)
        
        # Set window size
        self.resize(1280, 600)
        
    def on_layer_change(self, layer_name):
        self.visualizer = ActivationVisualizer(self.model_name, layer_name)
        
    def process_frame(self, frame):
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transform for model input
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.visualizer.device)
        
        # Get activations
        with torch.no_grad():
            self.visualizer.model(input_tensor)
        
        if self.visualizer.activations is None:
            return frame_rgb, None
            
        # Process activations
        activations = self.visualizer._normalize_activation(self.visualizer.activations[0])
        
        # Use half of total activations
        n_features = activations.shape[0] // 2
        n_cols = int(np.ceil(np.sqrt(n_features)))
        n_rows = int(np.ceil(n_features / n_cols))
        
        # Calculate feature size to match the display size
        feature_size = min(640 // n_cols, 480 // n_rows)
        activation_img = np.zeros((n_rows * feature_size, n_cols * feature_size, 3), dtype=np.uint8)
        
        for idx in range(n_features):
            i, j = idx // n_cols, idx % n_cols
            feature = activations[idx]
            feature = (feature * 255).astype(np.uint8)
            feature = cv2.resize(feature, (feature_size, feature_size))
            # Convert to grayscale
            feature_color = cv2.cvtColor(feature, cv2.COLOR_GRAY2BGR)
            activation_img[i*feature_size:(i+1)*feature_size, j*feature_size:(j+1)*feature_size] = feature_color
            
        return frame_rgb, activation_img
        
    def toggle_recording(self):
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_button.setText("Stop Recording")
            self.recorded_frames = []
        else:
            # Stop recording
            self.is_recording = False
            self.record_button.setText("Start Recording")
            self.save_recording()

    def save_recording(self):
        if not self.recorded_frames:
            return
            
        # Get the first frame to determine dimensions
        height, width = self.recorded_frames[0][0].shape[:2]  # Camera frame dimensions
        activation_height, activation_width = self.recorded_frames[0][1].shape[:2]  # Activation frame dimensions
        
        # Create video writer with combined width
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('recording.mp4', fourcc, 30.0, (width + activation_width, max(height, activation_height)))
        
        # Write frames
        for camera_frame, activation_frame in self.recorded_frames:
            # Create combined frame
            combined_frame = np.zeros((max(height, activation_height), width + activation_width, 3), dtype=np.uint8)
            
            # Place camera frame on the left
            combined_frame[:height, :width] = camera_frame
            
            # Place activation frame on the right
            combined_frame[:activation_height, width:width + activation_width] = activation_frame
            
            out.write(combined_frame)
            
        # Release video writer
        out.release()
        self.recorded_frames = []

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame
            frame_rgb, activation_img = self.process_frame(frame)
            
            # Store frame for recording if recording is active
            if self.is_recording and activation_img is not None:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                activation_bgr = cv2.cvtColor(activation_img, cv2.COLOR_RGB2BGR)
                self.recorded_frames.append((frame_bgr, activation_bgr))
            
            # Convert frames to QImage
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            if activation_img is not None:
                # Convert activation visualization to QImage
                height, width, channel = activation_img.shape
                bytes_per_line = 3 * width
                q_img = QImage(activation_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.activation_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                    self.activation_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
    def closeEvent(self, event):
        if self.is_recording:
            self.save_recording()
        self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    visualizer = CameraVisualizerGUI()
    visualizer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 