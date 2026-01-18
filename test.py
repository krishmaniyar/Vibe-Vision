import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from utils import detect_face_and_preprocess, draw_face_mesh, get_face_mesh_results

# Define emotion classes (must match training)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 48

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the same model architecture as in main.py
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Load the trained model
print("Loading model...")
model = EmotionCNN(NUM_CLASSES).to(device)
model.load_state_dict(torch.load('./model.pth', map_location=device))
model.eval()  # Set to evaluation mode
print("Model loaded successfully!")

# Initialize MediaPipe Face Detection (for preprocessing)
face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0,  # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5
)

# Initialize MediaPipe Face Mesh (for visualization)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam (try different indices if 0 doesn't work)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./files/Human_Face_Emotions_Video.mp4")

if not cap.isOpened():
    print("Error: Could not open webcam. Trying camera index 2...")
    cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nStarting real-time emotion detection...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect face and preprocess for emotion detection
    preprocessed_face, bbox = detect_face_and_preprocess(frame, face_detection)
    
    # Get face mesh results for visualization
    face_mesh_results = get_face_mesh_results(frame, face_mesh)
    
    if preprocessed_face is not None:
        # Convert to PyTorch tensor
        face_tensor = torch.FloatTensor(preprocessed_face).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(face_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
            emotion = EMOTIONS[predicted.item()]
            confidence_percent = confidence.item() * 100
        
        # Draw face mesh and label
        draw_face_mesh(frame, face_mesh_results, emotion, confidence_percent)
    else:
        # No face detected
        cv2.putText(frame, "No face detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display frame
    cv2.imshow('Real-time Emotion Detection', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Webcam released. Goodbye!")
