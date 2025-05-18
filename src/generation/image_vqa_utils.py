import json
from PIL import Image
import io
import numpy as np
import os

import cv2
import webcolors
from collections import Counter
import math
import itertools

# Parse labelme
def parse_labelme_json(json_path):
    # Util function to parse the labelme detection file. 
    with open(json_path, "r") as f:
        data = json.load(f)

    detected_objects = []
    for shape in data["shapes"]:
        if shape["shape_type"] == "rectangle":
            label = shape["label"]
            (x1, y1), (x2, y2) = shape["points"]
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            detected_objects.append({"label": label, "bbox": bbox})

    return detected_objects

# Object counting and presence
def count_objects(detected_objects, label):
    count = 0
    # Count the number of objects of a class given the class label
    for obj in detected_objects:
        if obj['label'] == label:
            count += 1
    return count

def has_object(detected_objects, label):
    # Check presence of an object given the label
    for obj in detected_objects:
      if obj['label'] == label:
        return True
    return False

# Object localization and grounding
def get_bbox_location(detected_objects, label):
    # Get all bounding box locations given the class label
    locs = []
    for obj in detected_objects:
      if obj['label'] == label:
        x1, y1, x2, y2 = obj['bbox']
        locs.append(f"[x1={x1:.0f}, y1={y1:.0f}, x2={x2:.0f}, y2={y2:.0f}]")
    return ", ".join(locs) if locs else "Not present"

def infer_class_at_bbox(detected_objects, bbox):
    for obj in detected_objects:
        if obj['bbox'] == bbox:
            return obj['label']

# Object description
def closest_color_name(rgb):
    #Find the closest CSS3 color name to the given RGB value."""
    min_dist = float("inf")
    closest_name = None
    for name, hex_val in webcolors.CSS3_NAMES_TO_HEX.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_val)
        dist = (r_c - rgb[0]) ** 2 + (g_c - rgb[1]) ** 2 + (b_c - rgb[2]) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def infer_color_of_object(image_path, detected_objects, label):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    for obj in detected_objects:
      if obj['label'] == label:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0:
                return "Unknown"         
            avg_color = np.mean(crop.reshape(-1, 3), axis=0).astype(int)

            try:
                name = webcolors.rgb_to_name(tuple(avg_color))
            except ValueError:
                name = closest_color_name(avg_color)
            return name
    return "Unknown"

# Surrounding Description
# def infer_time_of_day(image_path):
#     # Infer time of day from the image using a single rule
#     image = Image.open(image_path).convert("L")
#     brightness = np.mean(np.array(image))

#     if brightness < 50:
#         return "night"
#     else:
#         return "day"

def infer_time_of_day(image_path, night_thresh=70, day_thresh=100):
    """
    Infers time of day using filename hint first, then brightness if needed.

    Args:
        image_path (str): Path to the image.
        night_thresh (int): Max mean brightness for 'night'.
        day_thresh (int): Min mean brightness for 'day'.

    Returns:
        str: 'night', 'dusk/dawn', or 'day'
    """
    filename = os.path.basename(image_path).lower()

    # 1. Check filename hints
    if "Night" in filename:
        return "night"
    elif "Day" in filename:
        return "day"

    # 2. Fallback to luminance analysis
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    grayscale = np.dot(image_np[..., :3], [0.299, 0.587, 0.114])
    brightness = np.mean(grayscale)

    if brightness < night_thresh:
        return "night"
    elif brightness < day_thresh:
        return "dusk/dawn"
    else:
        return "day"


def compute_traffic_density(detected_objects, image_path):
    # Compute traffic density based on heuristics 
    vehicle_labels = ["car", "bus", "truck", "motorcycle", "autorickshaw"]
    vehicle_count = sum(count_objects(detected_objects, label) for label in vehicle_labels)
    try:
        image = Image.open(image_path)
        img_width, img_height = image.size
    except Exception:
        return "Unknown"

    img_area = img_width * img_height

    vehicle_area = 0
    for obj in detected_objects:
        if obj['label'] in vehicle_labels:
            x1, y1, x2, y2 = obj['bbox']
            vehicle_area += (x2 - x1) * (y2 - y1)
    vehicle_area_ratio = vehicle_area / img_area

    score = (vehicle_count * 0.5 + vehicle_area_ratio * 100 * 0.5)
    if score < 3:
        return "Low"
    elif score < 7: 
        return "Moderate"
    else:
        return "High"

def infer_scene_type(detected_objects, traffic_density):
    # Infer scene type
    has_pedestrian = has_object(detected_objects, "person")
    has_bike = has_object(detected_objects, "bike")
    has_auto = has_object(detected_objects, "autorickshaw")
    has_traffic_light = has_object(detected_objects, "traffic light")
    has_truck = has_object(detected_objects, "truck")

    if has_traffic_light and traffic_density in ["High", "Moderate"]:
        return "Urban"
    elif has_truck or has_auto or has_bike and not has_pedestrian and traffic_density in ["Low"]:
        return "Highway"
    elif has_auto and has_pedestrian and traffic_density in ["Low"]:
        return "Rural"
    else:
        return "Unknown"

# Spatial Relationships
def compute_iou(boxA, boxB):
    # Compute the IoU between two bounding boxes  
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def bbox_centroid(box):
    # Calculate centroid of the bounding box
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]

def determine_direction(centroid_a, centroid_b):
    # Use simple heuristic to translate into directional relationships 
    dx = centroid_b[0] - centroid_a[0]
    dy = centroid_b[1] - centroid_a[1]

    if abs(dx) > abs(dy):
        return "right of" if dx > 0 else "left of"
    else:
        return "behind" if dy > 0 else "ahead"

def euclidean_distance(p1, p2):
    # Calculate euclidean distance for two objects 
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_spatial_scene_graph(detected_objects, iou_threshold=0.1, near_distance_thresh=50):
    # Generate scenegraph based on IoU data
    graph = []
    for a, b in itertools.combinations(detected_objects, 2):
        label_a = a["label"]
        label_b = b["label"]

        bbox_a = a["bbox"]
        bbox_b = b["bbox"]
        centroid_a = bbox_centroid(bbox_a)
        centroid_b = bbox_centroid(bbox_b)
        iou = compute_iou(bbox_a, bbox_b)
        distance = euclidean_distance(centroid_a, centroid_b)

        if iou > iou_threshold:
            relation = "overlaps"
        elif distance < near_distance_thresh:
            relation = "near"
        else:
            relation = determine_direction(centroid_a, centroid_b)

        graph.append({
            "subject": label_a,
            "subject_bbox": bbox_a,
            "object": label_b,
            "object_bbox": bbox_b,
            "relation": relation
        })
    return graph




