import cv2
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from glob import glob

def detect_tubes(image_path):
    """
    Finds all microcentrifuge tube lids in an image.
    Returns list of (center_x, center_y, radius) for each detected circle.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=25,
        param1=50,
        param2=25,
        minRadius=10,
        maxRadius=40,
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype(int)
    return [(float(x), float(y), float(r)) for x, y, r in circles]


def estimate_angle(image_path, cx, cy, radius):
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    sample_r   = radius * 1.0
    num_samples = 72
    scores = []

    for i in range(num_samples):
        angle_deg = i * (360.0 / num_samples)
        angle_rad = math.radians(angle_deg)
        px = int(round(cx + sample_r * math.cos(angle_rad)))
        py = int(round(cy - sample_r * math.sin(angle_rad)))

        if px < 0 or py < 0 or px >= w or py >= h:
            scores.append((angle_deg, 0.0))
            continue

        scores.append((angle_deg, float(gray[py, px])))

    return max(scores, key=lambda t: t[1])[0]



def run_detection(images_folder, annotations_csv):
    
    df_gt       = pd.read_csv(annotations_csv)
    image_names = df_gt["image"].unique()

    rows = []
    found_count = 0

    for img_name in image_names:
        img_path = os.path.join(images_folder, img_name)
        if not os.path.exists(img_path):
            continue

        found_count += 1
        circles = detect_tubes(img_path)

        for (cx, cy, r) in circles:
            angle = estimate_angle(img_path, cx, cy, r)
            rows.append({
                "image":       img_name,
                "pred_x":      cx,
                "pred_y":      cy,
                "pred_radius": r,
                "pred_angle":  angle,
            })

    print(f"  Images found on disk : {found_count} / {len(image_names)}")
    print(f"  Total detections     : {len(rows)}")
    return pd.DataFrame(rows)

 
def evaluate(predictions_df, annotations_csv):
    df_gt = pd.read_csv(annotations_csv)
 
    total_matched   = 0
    total_missed    = 0
    total_extra     = 0
    angle_errors    = []
    position_errors = []
 
    DISTANCE_THRESHOLD = 20  # pixels upto which error can be considered correct
 
    for img_name in df_gt["image"].unique():
        gt_rows   = df_gt[df_gt["image"] == img_name]
        pred_rows = predictions_df[predictions_df["image"] == img_name]
 
        gt_list   = list(zip(gt_rows["center_x"], gt_rows["center_y"], gt_rows["angle_deg"]))
        pred_list = list(zip(pred_rows["pred_x"], pred_rows["pred_y"], pred_rows["pred_angle"]))
 
        if not pred_list:
            total_missed += len(gt_list)
            continue
 
        matched_preds = set()
 
        for gx, gy, ga in gt_list:
            # find the closest prediction to this ground truth tube
            best_dist, best_pi = float("inf"), -1
            for pi, (px, py, pa) in enumerate(pred_list):
                if pi in matched_preds:
                    continue
                d = math.hypot(gx - px, gy - py)
                if d < best_dist:
                    best_dist, best_pi = d, pi
 
            if best_pi >= 0 and best_dist <= DISTANCE_THRESHOLD:
                matched_preds.add(best_pi)
                position_errors.append(best_dist)
 
                pa = pred_list[best_pi][2]
                diff = abs(ga - pa) % 360
                angle_errors.append(diff if diff <= 180 else 360 - diff)
            else:
                total_missed += 1
 
        total_extra += len(pred_list) - len(matched_preds)
 
        total_matched += len(matched_preds)
 
    total_gt = len(df_gt)
 
    print("\n---- RESULTS ----")
    print(f"Total ground truth tubes : {total_gt}")
    print(f"  Tubes correctly found    : {total_matched}")
    print(f" Tubes missed             : {total_missed}")
    print(f"Extra (false detections) : {total_extra}")
    print(f"  Detection rate           : {total_matched / total_gt * 100:.1f}%")
    if position_errors:
        print(f" Avg position error       : {sum(position_errors) / len(position_errors):.1f} px")
    if angle_errors:
        print(f"Avg angle error          : {sum(angle_errors) / len(angle_errors):.1f} deg")
    print("-----------------")
 
 
if __name__ == "__main__":
    IMAGES_FOLDER = "images"
    ANNOTATIONS   = "annotations.csv"
 
    print("MICROCENTRIFUGE TUBE DETECTOR")
    print("=" * 40)
 
    print("\n[Step 1, 2, 3] Detecting tubes...")
    predictions = run_detection(IMAGES_FOLDER, ANNOTATIONS)
    predictions.to_csv("predictions.csv", index=False)
    print("  Predictions saved -> predictions.csv")
 
    print("\n[Step 4] Evaluating...")
    evaluate(predictions, ANNOTATIONS)