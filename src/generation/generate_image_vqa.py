import random
from image_vqa_utils import *  # Assuming all necessary helper functions are here
import os

from tqdm import tqdm
import csv

# Set the random seed for reproducibility
seed = 38
random.seed(seed)
np.random.seed(seed)

class VQAGenerator:
    def __init__(self, image_path, detection_file_path):
        self.image_path = image_path
        # Parse detected objects once during initialization
        self.detected_objects = parse_labelme_json(detection_file_path)
        # For convenience, alias shapes to detected_objects
        self.shapes = self.detected_objects

    def choose_random_class(self):
        # Chooses a random class label from the detected objects
        labels = [obj['label'] for obj in self.detected_objects if 'label' in obj]
        if not labels:
            return None
        return random.choice(labels)

    def answer_object_counting_questions(self):
        # Pick two random classes for comparison
        class_a = self.choose_random_class()
        class_b = self.choose_random_class()

        # If no classes found, return empty list
        if class_a is None or class_b is None:
            return []

        return [
            {
                "question": f"How many objects of type {class_a} are in the image?",
                "answer": count_objects(self.shapes, class_a)
            },
            {
                "question": f"Is there a {class_a} present in the image?",
                "answer": "Yes" if has_object(self.shapes, class_a) else "No"
            },
            {
                "question": f"Are there more {class_a}'s than {class_b}'s?",
                "answer": "Yes" if count_objects(self.shapes, class_a) > count_objects(self.shapes, class_b) else "No"
            }
        ]

    def answer_object_localization_questions(self):
        class_a = self.choose_random_class()
        if class_a is None:
            return []

        return [
            {
                "question": f"Where is the {class_a} present in the image?",
                "answer": get_bbox_location(self.detected_objects, class_a)
            }
        ]

    def answer_object_description_questions(self):
        class_a = self.choose_random_class()
        if class_a is None:
            return []

        # Placeholder bbox for infer_class_at_bbox - you need to define how to get it
        # For now, let's pick the bbox of the first object of class_a
        bbox = None
        for obj in self.detected_objects:
            if obj.get('label') == class_a:
                bbox = obj.get('bbox')
                break

        return [
            {
                "question": f"What is the color of the {class_a} in the image?",
                "answer": infer_color_of_object(self.image_path, self.detected_objects, class_a)
            },
            {
                "question": f"What class is the object at bounding box {bbox}?",
                "answer": infer_class_at_bbox(self.detected_objects, bbox=bbox) if bbox else "Unknown"
            }
        ]

    def answer_surrounding_description_questions(self):
        traffic_density = compute_traffic_density(self.detected_objects, self.image_path)
        return [
            {
                "question": "What time of day is it?",
                "answer": infer_time_of_day(self.image_path)
            },
            {
                "question": "What is the traffic density?",
                "answer": traffic_density
            },  
        ]

    def answer_spatial_relationship_questions_from_graph(self):
        spatial_relations = generate_spatial_scene_graph(self.shapes)

        return [
            {
                "question": f"What is the spatial relationship between {rel['subject']} and {rel['object']}?",
                "answer": f"{rel['subject']} is {rel['relation']} {rel['object']}"
            }
            for rel in spatial_relations
        ]


    def answer_spatial_relationship_questions_from_graph(self):
        spatial_relations = generate_spatial_scene_graph(self.shapes)
        if not spatial_relations:
            return []
        rel = random.choice(spatial_relations)

        bbox_subject = rel['subject_bbox']
        bbox_object = rel['object_bbox']

        question = (
            f"What is the spatial relationship between {rel['subject']} "
            f"(bbox: {bbox_subject}) and {rel['object']} (bbox: {bbox_object})?"
        )

        answer = f"{rel['subject']} is {rel['relation']} {rel['object']}"

        return [{
            "question": question,
            "answer": answer
        }]


    def answer_all_questions(self, srl=None):
        # srl is optional and only used for risk analysis
        result = {
            "Object Counting": self.answer_object_counting_questions(),
            "Object Localization": self.answer_object_localization_questions(),
            "Object Description": self.answer_object_description_questions(),
            "Surrounding Description": self.answer_surrounding_description_questions(),
            "Spatial Relationship": self.answer_spatial_relationship_questions_from_graph(),
        }
            #result["Risk Analysis"] = self.answer_risk_analysis_questions(srl)
        return result

def process_directory(image_dir, output_csv):
    rows = []

    for root, _, files in os.walk(image_dir):
        for file in tqdm(files):
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                json_path = os.path.splitext(image_path)[0] + ".json"

                if not os.path.exists(json_path):
                    continue

                try:
                    vqa_generator = VQAGenerator(image_path, json_path)
                    all_qas = vqa_generator.answer_all_questions()

                    for category, qa_list in all_qas.items():
                        for qa in qa_list:
                            rows.append({
                                "filename": os.path.basename(image_path),
                                "category": category,
                                "question": qa["question"],
                                "answer": qa["answer"]
                            })

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "category", "question", "answer"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    image_dir = r"dataset/image_data/images/train"
    output_csv = "dataset/vqa_dataset_train.csv"
    process_directory(image_dir, output_csv)
    print(f"VQA results saved to {output_csv}")
