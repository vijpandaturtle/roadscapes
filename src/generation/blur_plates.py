import argparse
import os
import cv2
import yolov5
from collections import defaultdict
from datetime import datetime


class PlateBlurrer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        # Load YOLOv5 license plate model from Hugging Face
        self.model = yolov5.load('keremberke/yolov5m-license-plate')
        self.processed_files = defaultdict(list)
        
        # Create a timestamp for the log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(folder_path, f"blur_summary_{timestamp}.txt")

    def process_folder(self):
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith(".jpg"):
                    image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, self.folder_path)
                    
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Could not load {image_path}")
                        continue

                    results = self.model(image_path)
                    predictions = results.pred[0]
                    plates_found = len(predictions)
                    
                    for box in predictions:
                        x_min, y_min, x_max, y_max = map(int, box[:4])
                        # Ensure coordinates are within image bounds
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(img.shape[1], x_max)
                        y_max = min(img.shape[0], y_max)
                        # Blur the detected license plate region
                        roi = img[y_min:y_max, x_min:x_max]
                        if roi.size > 0:
                            roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
                            img[y_min:y_max, x_min:x_max] = roi_blur

                    # Save the anonymized image
                    cv2.imwrite(image_path, img)
                    self.processed_files[relative_path].append({
                        'filename': file,
                        'plates_found': plates_found
                    })
                    print(f"Processed {file}: found {plates_found} plates")

    def save_summary(self):
        total_images = 0
        total_plates = 0
        
        with open(self.log_file, 'w') as f:
            f.write("License Plate Blurring Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base Directory: {self.folder_path}\n")
            f.write("=" * 50 + "\n\n")
            
            for folder, files in sorted(self.processed_files.items()):
                if folder == '.':
                    f.write("\nRoot folder:\n")
                else:
                    f.write(f"\nFolder: {folder}\n")
                f.write("-" * 50 + "\n")
                
                for file_info in sorted(files, key=lambda x: x['filename']):
                    f.write(f"- {file_info['filename']} "
                           f"(plates found: {file_info['plates_found']})\n")
                    total_images += 1
                    total_plates += file_info['plates_found']
            
            f.write("\nSummary:\n")
            f.write(f"Total images processed: {total_images}\n")
            f.write(f"Total plates detected and blurred: {total_plates}\n")
        
        print(f"\nSummary has been saved to: {self.log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv5 license plate blurring for anonymization."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to folder with images."
    )
    args = parser.parse_args()

    blurrer = PlateBlurrer(args.folder)
    blurrer.process_folder()
    blurrer.save_summary()
