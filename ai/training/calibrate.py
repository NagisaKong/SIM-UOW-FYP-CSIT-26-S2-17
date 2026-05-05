from ai.face_recognizer import ArcFaceRecognizer 
from ai.config import AIConfig
from sklearn.metrics import precision_recall_curve
import numpy as np
import cv2 # Required for image padding

def calibrate_threshold(synthetic_pairs, labels):
    scores = []
    valid_labels = [] # Keep track of labels for faces we ACTUALLY detect
    
    cfg = AIConfig()
    arcface = ArcFaceRecognizer(cfg)

    print(f"Starting calibration on {len(synthetic_pairs)} pairs...")

    # Use zip() to loop through pairs and labels simultaneously
    for (img_a, img_b), label in zip(synthetic_pairs, labels):
        
        # FIX 1: Shrink and pad the StyleGAN image with a black border.
        # This tricks the webcam-trained face detector into easily finding the face.
        img_a = cv2.copyMakeBorder(cv2.resize(img_a, (512, 512)), 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0,0,0])
        img_b = cv2.copyMakeBorder(cv2.resize(img_b, (512, 512)), 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0,0,0])

        faces_a = arcface.app.get(img_a)
        faces_b = arcface.app.get(img_b)
        
        if not faces_a or not faces_b:
            print("Warning: Face detector failed to find a face. Skipping pair.")
            continue
            
        feat_a = faces_a[0].normed_embedding
        feat_b = faces_b[0].normed_embedding
        
        similarity = np.dot(feat_a, feat_b) / (np.linalg.norm(feat_a) * np.linalg.norm(feat_b))
        
        # FIX 2: Only add the label if the score was actually calculated!
        scores.append(similarity)
        valid_labels.append(label)

    # Safety net: If something goes completely wrong, return the default threshold
    if not scores:
        print("ERROR: No faces detected at all. Defaulting threshold to 0.35")
        return 0.35

    # Calculate optimal Equal Error Rate (EER) threshold
    precision, recall, thresholds = precision_recall_curve(valid_labels, scores)
    
    # Calculate F1 score to find the optimal balance (added 1e-8 to prevent division by zero)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    optimal_idx = np.argmax(f1_scores)
    # Ensure index doesn't go out of bounds of the thresholds array
    optimal_idx = min(optimal_idx, len(thresholds) - 1)
    
    return float(thresholds[optimal_idx])