import cv2
import mediapipe as mp
import numpy as np

IMG_SIZE = 48  # Standard size for emotion detection models


def get_face_landmarks(image, draw=False, static_image_mode=True):
    """Legacy function for face landmarks extraction"""
    # Read the input image
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                max_num_faces=1,
                                                min_detection_confidence=0.5)
    image_rows, image_cols, _ = image.shape
    results = face_mesh.process(image_input_rgb)

    image_landmarks = []

    if results.multi_face_landmarks:

        if draw:

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        ls_single_face = results.multi_face_landmarks[0].landmark
        xs_ = []
        ys_ = []
        zs_ = []
        for idx in ls_single_face:
            xs_.append(idx.x)
            ys_.append(idx.y)
            zs_.append(idx.z)
        for j in range(len(xs_)):
            image_landmarks.append(xs_[j] - min(xs_))
            image_landmarks.append(ys_[j] - min(ys_))
            image_landmarks.append(zs_[j] - min(zs_))

    return image_landmarks


def detect_face_and_preprocess(frame, face_detector):
    """
    Detect face in frame and preprocess it for emotion detection model.
    
    Args:
        frame: Input BGR frame from webcam
        face_detector: MediaPipe FaceDetection object
    
    Returns:
        preprocessed_face: Preprocessed face image ready for model (48x48 grayscale, normalized)
        face_bbox: Bounding box coordinates (x, y, w, h) or None if no face detected
    """
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = face_detector.process(rgb_frame)
    
    if results.detections:
        # Get the first face detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w, _ = frame.shape
        
        # Convert relative coordinates to absolute
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        face_w = int(bbox.width * w)
        face_h = int(bbox.height * h)
        
        # Add some padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        face_w = min(w - x, face_w + 2 * padding)
        face_h = min(h - y, face_h + 2 * padding)
        
        # Extract face region
        face_roi = frame[y:y+face_h, x:x+face_w]
        
        if face_roi.size > 0:
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Resize to model input size
            resized_face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
            
            # Normalize to [0, 1]
            normalized_face = resized_face.astype('float32') / 255.0
            
            # Reshape for PyTorch: (1, 1, 48, 48) - batch, channels, height, width
            preprocessed_face = normalized_face.reshape(1, 1, IMG_SIZE, IMG_SIZE)
            
            return preprocessed_face, (x, y, face_w, face_h)
    
    return None, None


def get_face_mesh_results(frame, face_mesh):
    """
    Get face mesh results from MediaPipe for visualization.
    
    Args:
        frame: Input BGR frame from webcam
        face_mesh: MediaPipe FaceMesh object
    
    Returns:
        face_mesh_results: MediaPipe face mesh results or None
    """
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame for face mesh
    results = face_mesh.process(rgb_frame)
    
    return results if results.multi_face_landmarks else None


def draw_face_mesh(frame, face_mesh_results, emotion, confidence):
    """
    Draw face mesh and emotion label on frame.
    
    Args:
        frame: Input frame
        face_mesh_results: MediaPipe face mesh results
        emotion: Emotion label string
        confidence: Confidence score
    """
    if not face_mesh_results or not face_mesh_results.multi_face_landmarks:
        return
    
    mp_drawing = mp.solutions.drawing_utils
    
    # Custom drawing specs with white/light grey colors
    # White color in BGR format: (255, 255, 255)
    # Light grey color in BGR format: (200, 200, 200)
    white_connection_spec = mp_drawing.DrawingSpec(
        color=(255, 255, 255),  # White in BGR
        thickness=1,
        circle_radius=0
    )
    
    light_grey_connection_spec = mp_drawing.DrawingSpec(
        color=(200, 200, 200),  # Light grey in BGR
        thickness=1,
        circle_radius=0
    )
    
    # Draw face mesh
    for face_landmarks in face_mesh_results.multi_face_landmarks:
        # Draw face mesh contours with white color
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,  # Don't draw landmark points
            connection_drawing_spec=white_connection_spec
        )
        
        # Draw face mesh tesselation with light grey color
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,  # Don't draw landmark points
            connection_drawing_spec=light_grey_connection_spec
        )
        
        # Get face bounding box from landmarks for label placement
        h, w, _ = frame.shape
        xs = [landmark.x * w for landmark in face_landmarks.landmark]
        ys = [landmark.y * h for landmark in face_landmarks.landmark]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        
        # Prepare text
        label = f"{emotion} ({confidence:.1f}%)"
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text (above the face) - using white/light grey instead of green
        text_y = max(0, y_min - text_height - 10)
        cv2.rectangle(frame, (x_min, text_y), 
                      (x_min + text_width, text_y + text_height + 5), (200, 200, 200), -1)
        
        # Draw text in black for contrast
        cv2.putText(frame, label, (x_min, text_y + text_height), 
                    font, font_scale, (0, 0, 0), thickness)


def draw_face_bbox(frame, bbox, emotion, confidence):
    """
    Draw bounding box and emotion label on frame (legacy function).
    
    Args:
        frame: Input frame
        bbox: Bounding box (x, y, w, h)
        emotion: Emotion label string
        confidence: Confidence score
    """
    if bbox is None:
        return
    
    x, y, w, h = bbox
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Prepare text
    label = f"{emotion} ({confidence:.1f}%)"
    
    # Get text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Draw background rectangle for text
    cv2.rectangle(frame, (x, y - text_height - 10), 
                  (x + text_width, y), (0, 255, 0), -1)
    
    # Draw text
    cv2.putText(frame, label, (x, y - 5), 
                font, font_scale, (0, 0, 0), thickness)