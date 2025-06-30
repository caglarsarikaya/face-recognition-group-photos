import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Module level initialization - one time only
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(face_img):
    # Try to get embedding from the cropped face
    try:
        # Resize to a standard size for more consistent results
        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
            # Ensure the image is in the right format for the model
            resized_face = cv2.resize(face_img, (112, 112))
            
            # Use the app to get embedding directly
            # The model should recognize the face without detection since it's pre-cropped
            faces = app.get(resized_face)
            if faces:
                print("    Successfully extracted embedding from cropped face")
                return faces[0].embedding
            else:
                print("    No face found in the cropped image")
    except Exception as e:
        print(f"    Error processing cropped face: {str(e)}")
    
    # Original method as fallback
    try:
        faces = app.get(face_img)
        if faces:
            print("    Successfully extracted embedding using original method")
            return faces[0].embedding
    except Exception as e:
        print(f"    Fallback method failed: {str(e)}")
    
    print("    Failed to extract embedding with both methods")
    return None

def load_reference_embeddings(reference_dir):
    # Global app değişkenini kullan
    db = {}
    print(f"\nLoading reference embeddings from: {reference_dir}")
    
    for label in os.listdir(reference_dir):
        label_dir = os.path.join(reference_dir, label)
        if not os.path.isdir(label_dir):
            continue
            
        print(f"  Processing reference label: {label}")
        embeddings = []
        
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            print(f"    Loading image: {img_path}")
            
            img = cv2.imread(img_path)
            if img is not None:
                print(f"    Image loaded, shape: {img.shape}")
                faces = app.get(img)
                if faces:
                    print(f"    Found {len(faces)} face(s) in reference image")
                    embeddings.append(faces[0].embedding)
                else:
                    print(f"    WARNING: No faces detected in reference image: {img_path}")
            else:
                print(f"    ERROR: Could not load image: {img_path}")
                
        if embeddings:
            db[label] = np.stack(embeddings)
            print(f"  Added {len(embeddings)} embedding(s) for {label}")
        else:
            print(f"  WARNING: No valid embeddings found for {label}")
            
    return db 