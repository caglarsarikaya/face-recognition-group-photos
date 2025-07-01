import os
import numpy as np
from face_detection.detector import detect_faces

class DataLoader:
    """Handles loading of reference embeddings and photo files"""
    
    def __init__(self, reference_dir='data/reference_faces/', photos_dir='data/photos'):
        self.reference_dir = reference_dir
        self.photos_dir = photos_dir
    
    def load_reference_embeddings(self):
        """Load reference faces and their embeddings directly using the detector"""
        db = {}
        print(f"\nLoading reference embeddings from: {self.reference_dir}")
        
        for label in os.listdir(self.reference_dir):
            label_dir = os.path.join(self.reference_dir, label)
            if not os.path.isdir(label_dir):
                continue
                
            print(f"  Processing reference label: {label}")
            embeddings = []
            
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                print(f"    Loading image: {img_path}")
                
                # Use the detector to get embeddings directly
                _, _, face_embeddings = detect_faces(img_path)
                
                if face_embeddings and len(face_embeddings) > 0:
                    print(f"    Found {len(face_embeddings)} face(s) with embeddings")
                    # Take the first face embedding (assuming reference images have one primary face)
                    embeddings.append(face_embeddings[0])
                else:
                    print(f"    WARNING: No face embeddings found in reference image: {img_path}")
                    
            if embeddings:
                db[label] = np.stack(embeddings)
                print(f"  Added {len(embeddings)} embedding(s) for {label}")
            else:
                print(f"  WARNING: No valid embeddings found for {label}")
                
        return db
    
    def get_photo_files(self):
        """Get list of photo files to process"""
        photo_files = [
            os.path.join(self.photos_dir, f) 
            for f in os.listdir(self.photos_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        print(f"\nFound {len(photo_files)} photo files to process")
        return photo_files 