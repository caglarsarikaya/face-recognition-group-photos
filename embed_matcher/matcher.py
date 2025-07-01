import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def match_embedding(embedding, reference_db, threshold=0.5):
    best_label = None
    best_score = -1
    for label, embs in reference_db.items():
        # Compute cosine similarity with all reference embeddings for this label
        sims = cosine_similarity([embedding], embs)[0]
        score = np.max(sims)
        if score > best_score:
            best_score = score
            best_label = label
            
    # Eğer en iyi eşleşme bile eşik değerinden düşükse, None döndür
    if best_score < threshold:
        return None, best_score
    return best_label, best_score 