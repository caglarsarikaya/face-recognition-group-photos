from data_loader import DataLoader
from output_manager import OutputManager
from face_processor import FaceProcessor

def main():
    
    # Step 1: Load data
    print("=== STEP 1: LOADING DATA ===")
    data_loader = DataLoader()
    reference_db = data_loader.load_reference_embeddings()
    photo_files = data_loader.get_photo_files()
    
    # Step 2: Initialize processors
    print("=== STEP 2: INITIALIZING PROCESSORS ===")
    face_processor = FaceProcessor(reference_db, threshold=0.5)
    output_manager = OutputManager()
    
    # Step 3: Display reference database info
    face_processor.print_reference_database_info()
    
    # Step 4: Create output folders
    print("=== STEP 3: CREATING OUTPUT STRUCTURE ===")
    output_manager.create_output_folders(reference_db.keys())
    
    # Step 5: Process each photo
    print("=== STEP 4: PROCESSING PHOTOS ===")
    for photo_path in photo_files:
        # Detect and recognize faces in the photo
        recognized_labels = face_processor.process_photo(photo_path)
        
        # Copy photo to appropriate output folders
        for label in recognized_labels:
            output_manager.copy_photo_to_label_folder(photo_path, label)
    
    print("\n=== PROCESSING COMPLETE ===")

if __name__ == '__main__':
    main() 