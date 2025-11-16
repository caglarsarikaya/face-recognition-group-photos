import os
import shutil

class OutputManager:
    """Handles output folder creation and file copying"""
    
    def __init__(self, output_dir='data/output'):
        self.output_dir = output_dir
    
    def create_output_folders(self, reference_labels):
        """Create output folders for each reference label"""
        print(f"\nCreating output folders in: {self.output_dir}")
        created_folders = []
        
        for label in reference_labels:
            label_dir = os.path.join(self.output_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            created_folders.append(label_dir)
            print(f"  Created output folder: {label_dir}")
            
        return created_folders
    
    def copy_photo_to_label_folder(self, photo_path, label):
        """Copy a photo to the appropriate label folder"""
        label_dir = os.path.join(self.output_dir, label)
        output_path = os.path.join(label_dir, os.path.basename(photo_path))
        shutil.copy2(photo_path, output_path)
        print(f"  Copied {photo_path} to {output_path}")
        return output_path 