import os
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, recall_score, precision_score
import pandas as pd
import cv2

def load_image(path, mode='L'):
    """Read image and convert to numpy array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert(mode)  # 'L' for grayscale, 'RGB' for color
    return np.array(img)

def binarize_image(img, threshold):
    """Binarize image based on threshold."""
    return (img >= threshold).astype(np.uint8)

def apply_micro_branch(image, prob_map, brightness_threshold=100/255.0):
    """Apply Micro Branch to refine probability map."""
    # Convert image to grayscale if RGB
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0
    else:
        gray = image / 255.0
    
    # Create binary mask (1 for dark pixels, 0 for bright)
    mask = (gray < brightness_threshold).astype(np.float32)
    
    # Refine probability map (element-wise multiplication)
    refined_prob = prob_map * mask
    
    return refined_prob

def calculate_metrics(pred, gt, ignore_label=255):
    """Calculate F1-score, recall, precision, ignoring ignore_label."""
    valid_mask = gt != ignore_label
    pred = pred[valid_mask].flatten()
    gt = gt[valid_mask].flatten()
    if len(pred) == 0 or len(gt) == 0:
        return 0.0, 0.0, 0.0
    f1 = f1_score(gt, pred, average='binary', pos_label=1)
    recall = recall_score(gt, pred, average='binary', pos_label=1)
    precision = precision_score(gt, pred, average='binary', pos_label=1)
    return f1, recall, precision

def evaluate_dataset(test_list_path, pred_dir, gt_dir, img_dir, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9], brightness_threshold=100/255.0):
    """Evaluate on the entire test set with Micro Branch."""
    results = []
    with open(test_list_path, 'r') as f:
        lines = f.readlines()
    
    for threshold in thresholds:
        f1_scores, recalls, precisions = [], [], []
        for line in lines:
            img_name, gt_name = line.strip().split()
            base_name = os.path.basename(img_name).split('.')[0]
            pred_path = os.path.join(pred_dir, f"{base_name}_pred.png")
            gt_path = os.path.join(gt_dir, gt_name)
            img_path = os.path.join(img_dir, img_name)  # Load original image
            
            # Debug: In đường dẫn để kiểm tra
            print(f"Loading prediction: {pred_path}")
            print(f"Loading ground truth: {gt_path}")
            print(f"Loading original image: {img_path}")
            
            # Load prediction, ground truth, and original image
            pred = load_image(pred_path, mode='L') / 255.0  # Normalize to [0, 1]
            gt = load_image(gt_path, mode='L')
            img = load_image(img_path, mode='RGB')  # Load RGB for Micro Branch
            
            # Apply Micro Branch
            refined_pred = apply_micro_branch(img, pred, brightness_threshold)
            
            # Binarize refined prediction
            pred_bin = binarize_image(refined_pred, threshold)
            
            # Calculate metrics
            f1, recall, precision = calculate_metrics(pred_bin, gt)
            f1_scores.append(f1)
            recalls.append(recall)
            precisions.append(precision)
        
        # Calculate mean Macro F1-score
        mean_f1 = np.mean(f1_scores)
        mean_recall = np.mean(recalls)
        mean_precision = np.mean(precisions)
        results.append({
            'threshold': threshold,
            'f1_score': mean_f1,
            'recall': mean_recall,
            'precision': mean_precision
        })
    
    return results

def save_results_to_csv(results, output_csv):
    """Save results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

# Configuration
test_list_path = '/kaggle/working/deeplab-pytorch/data/val.txt'
pred_dir = '/kaggle/working/deeplab-pytorch/outputs/predictions/crack_detection/deeplabv2_resnet101_msc/val'
gt_dir = '/kaggle/working/weakly-sup-crackdet/models/deeplab/research/deeplab/datasets/data'
img_dir = '/kaggle/working/weakly-sup-crackdet/models/deeplab/research/deeplab/datasets/data'
output_csv = '/kaggle/working/deeplab_results_with_mib.csv'
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
brightness_threshold = 100/255.0

# Debug: Kiểm tra thư mục pred_dir
if not os.path.exists(pred_dir):
    print(f"Prediction directory does not exist: {pred_dir}")
else:
    print(f"Prediction directory contents: {os.listdir(pred_dir)}")

# Evaluate
results = evaluate_dataset(test_list_path, pred_dir, gt_dir, img_dir, thresholds, brightness_threshold)
save_results_to_csv(results, output_csv)
print(f"Results saved to {output_csv}")