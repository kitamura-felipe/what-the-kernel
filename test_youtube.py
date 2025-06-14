from what_the_kernel.camera_visualizer import CameraVisualizerGUI
from PyQt5.QtWidgets import QApplication
import sys
import yt_dlp
import os
import cv2

class YouTubeVisualizerGUI(CameraVisualizerGUI):
    def __init__(self, youtube_url, model_name="resnet18"):
        # Download the YouTube video
        print(f"Downloading video from {youtube_url}...")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]',  # Best quality MP4
            'outtmpl': 'temp_video.mp4',  # Output filename
            'quiet': True,  # Suppress output
            'no_warnings': True  # Suppress warnings
        }
        
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # Initialize parent class
        super().__init__(model_name)
        
        # Replace camera capture with video file
        self.cap.release()  # Release the camera
        self.cap = cv2.VideoCapture("temp_video.mp4")
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video file")
            
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def closeEvent(self, event):
        if self.is_recording:
            self.save_recording()
        self.cap.release()
        # Clean up the temporary video file
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
        event.accept()

def main():
    app = QApplication(sys.argv)
    # Replace this URL with your desired YouTube video URL
    youtube_url = "https://www.youtube.com/watch?v=39ColarOWKo" # "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    visualizer = YouTubeVisualizerGUI(youtube_url)
    visualizer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 