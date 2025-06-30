import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Initialize the face analysis model once globally
face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def detect_faces(image_path):
    print(f"  Detecting faces in: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: Could not read image: {image_path}")
        return [], [], []
        
    print(f"  Image loaded, shape: {image.shape}")
    
    # Use InsightFace to detect and recognize faces in one step
    faces = face_app.get(image)
    
    bboxes = []
    embeddings = []
    cropped_faces = []
    
    if faces and len(faces) > 0:
        print(f"  Found {len(faces)} face(s)")
        for i, face in enumerate(faces):
            # Get bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            bboxes.append((x1, y1, x2, y2))
            
            # Store the embedding
            embeddings.append(face.embedding)
            
            # Crop the face for visualization if needed
            cropped = image[y1:y2, x1:x2]
            cropped_faces.append(cropped)
            
            print(f"  Face #{i}: bbox=({x1},{y1},{x2},{y2}), embedding shape={face.embedding.shape}")
    else:
        print("  No faces detected")
        
    return bboxes, cropped_faces, embeddings 