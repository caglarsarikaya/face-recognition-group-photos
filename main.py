from face_detection.detector import detect_faces
from matcher.matcher import match_embedding
import cv2
import os
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_reference_embeddings(reference_dir):
    """Load reference faces and their embeddings directly using the detector"""
    from face_detection.detector import detect_faces
    
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

def main():
    photos_dir = 'data/photos'
    photo_files = [os.path.join(photos_dir, f) for f in os.listdir(photos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # Load reference embeddings
    reference_db = load_reference_embeddings('data/reference_faces/')
    
    # Debug: Print reference database info
    print("\n=== REFERENCE DATABASE ===")
    for label, embs in reference_db.items():
        print(f"Label: {label}, Number of references: {len(embs)}")
    print("========================\n")

    # Create output folders matching reference faces
    for label in reference_db.keys():
        label_dir = os.path.join('output', label)
        os.makedirs(label_dir, exist_ok=True)
        print(f"Created output folder: {label_dir}")

    # For each photo
    for photo_path in photo_files:
        print(f"\nProcessing: {photo_path}")
        faces_bbox, cropped_faces, face_embeddings = detect_faces(photo_path)
        recognized_labels = set()
        
        # Process each detected face
        for i, (bbox, face_img, embedding) in enumerate(zip(faces_bbox, cropped_faces, face_embeddings)):
            # Debug: Calculate similarity directly with all reference embeddings
            print(f"  Face #{i} - Checking similarities:")
            for label, ref_embs in reference_db.items():
                sims = cosine_similarity([embedding], ref_embs)[0]
                max_sim = np.max(sims)
                print(f"    - {label}: max similarity = {max_sim:.4f}")
            
            # Regular matching
            label, score = match_embedding(embedding, reference_db, threshold=0.5)
            print(f"  {os.path.basename(photo_path)} - Face #{i}: {label if label else 'Unknown'} (score={score:.4f})")
            
            if label is not None:
                recognized_labels.add(label)
        
        # Copy to output/{label}/ for each recognized label
        for label in recognized_labels:
            label_dir = os.path.join('output', label)
            output_path = os.path.join(label_dir, os.path.basename(photo_path))
            shutil.copy2(photo_path, output_path)
            print(f"  Copied {photo_path} to {output_path}")

if __name__ == '__main__':
    main() 