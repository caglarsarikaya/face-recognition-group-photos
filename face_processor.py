import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from face_detection.detector import detect_faces
from embed_matcher.matcher import match_embedding

class FaceProcessor:
    """Handles face detection and recognition processing"""
    
    def __init__(self, reference_db, threshold=0.5):
        self.reference_db = reference_db
        self.threshold = threshold
    
    def process_photo(self, photo_path):
        """Process a single photo and return recognized labels"""
        print(f"\nProcessing: {photo_path}")
        faces_bbox, cropped_faces, face_embeddings = detect_faces(photo_path)
        recognized_labels = set()
        
        # Process each detected face
        for i, (bbox, face_img, embedding) in enumerate(zip(faces_bbox, cropped_faces, face_embeddings)):
            self._print_face_similarities(i, embedding)
            
            # Regular matching
            label, score = match_embedding(embedding, self.reference_db, threshold=self.threshold)
            print(f"  {os.path.basename(photo_path)} - Face #{i}: {label if label else 'Unknown'} (score={score:.4f})")
            
            if label is not None:
                recognized_labels.add(label)
        
        return recognized_labels
    
    def _print_face_similarities(self, face_index, embedding):
        """Print similarity scores for debugging"""
        print(f"  Face #{face_index} - Checking similarities:")
        for label, ref_embs in self.reference_db.items():
            sims = cosine_similarity([embedding], ref_embs)[0]
            max_sim = np.max(sims)
            print(f"    - {label}: max similarity = {max_sim:.4f}")
    
    def print_reference_database_info(self):
        """Print information about the reference database"""
        print("\n=== REFERENCE DATABASE ===")
        for label, embs in self.reference_db.items():
            print(f"Label: {label}, Number of references: {len(embs)}")
        print("========================\n") 