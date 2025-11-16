# Face Detection and Recognition System

A Python-based face detection and recognition system that can identify specific people in photos using reference face embeddings. This project uses state-of-the-art deep learning models to achieve high accuracy in face recognition tasks.

## 📊 Dataset Source

The test and training images used in this project are from the [Friends Character Face Recognition Dataset](https://www.kaggle.com/datasets/amiralikalbasi/images-of-friends-character-for-face-recognition) on Kaggle.

## 🏗️ Project Architecture

This project follows SOLID principles with a clean, modular architecture:

```
📁 face-detection/
├── 📄 main.py              # 🎯 Main orchestration script
├── 📄 data_loader.py       # 📂 Handles data loading operations
├── 📄 face_processor.py    # 🔍 Processes face detection and recognition
├── 📄 output_manager.py    # 📁 Manages output file operations
├── 📁 face_detection/      # 🎭 Face detection module (InsightFace)
├── 📁 embed_matcher/       # 🔗 Embedding matching module
├── 📁 data/                # 📊 Data storage
│   ├── 📁 photos/          # 📸 Photos to analyze
│   └── 📁 reference_faces/ # 👤 Reference faces for each person
└── 📁 output/              # 📤 Results organized by person
```

## 🧠 How It Works

### Core Technology Stack

- **InsightFace**: State-of-the-art face recognition model that provides unified face detection and embedding generation
- **OpenCV**: Image processing and computer vision operations
- **scikit-learn**: Cosine similarity calculations for face matching
- **NumPy**: Numerical computations and array operations

### Process Flow

1. **Data Loading**: Loads reference face images and creates embeddings for each person
2. **Face Detection**: Processes target photos to detect all faces present
3. **Embedding Generation**: Creates 512-dimensional feature vectors for each detected face
4. **Similarity Matching**: Compares face embeddings using cosine similarity
5. **Result Organization**: Copies photos to appropriate output folders based on recognized faces

### Face Recognition Process

The system uses a **unified approach** with InsightFace, which means:
- **Single Model**: One algorithm handles both detection and recognition
- **Consistent Embeddings**: Same feature space for all operations
- **High Accuracy**: State-of-the-art performance on face recognition tasks
- **No Library Conflicts**: Eliminates compatibility issues between different face processing libraries

## 📁 Folder Structure

### Data Organization

- **`data/photos/`**: Place all photos you want to analyze here
  - Supports: `.jpg`, `.jpeg`, `.png`, `.bmp` formats
  - Can contain multiple people in each photo
  - Photos will be processed to find all faces

- **`data/reference_faces/`**: Reference images for each person you want to recognize
  - Create subfolders for each person (e.g., `Chandler/`, `Joey/`, `Monica/`)
  - Place multiple photos of each person in their respective folders
  - More reference images = better recognition accuracy

### Output Structure

- **`output/`**: Results organized by recognized person
  - Each person gets their own folder (e.g., `output/Chandler/`)
  - Photos containing that person are copied to their folder
  - One photo can appear in multiple folders if it contains multiple recognized people

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd face-recognition-group-photos
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 How to Run

### Basic Usage

1. **Prepare your data**:
   - Add photos to analyze in `data/photos/`
   - Add reference faces in `data/reference_faces/[PersonName]/`

2. **Run the face recognition system**:
   ```bash
   python main.py
   ```

3. **Check results**:
   - Look in the `output/` folder for organized results
   - Each person's folder contains photos where they were recognized

### Example Output

When the system processes photos, you'll see detailed output like this:

```
Processing: data/photos\all (9).jpg
  Detecting faces in: data/photos\all (9).jpg
  Image loaded, shape: (720, 1280, 3)
  Found 6 face(s)
  Face #0: bbox=(1068,269,1169,414), embedding shape=(512,)
  Face #1: bbox=(166,267,263,400), embedding shape=(512,)
  Face #2: bbox=(328,54,436,200), embedding shape=(512,)
  Face #3: bbox=(593,338,695,474), embedding shape=(512,)
  Face #4: bbox=(859,215,944,347), embedding shape=(512,)
  Face #5: bbox=(500,103,586,223), embedding shape=(512,)
  
  Face #0 - Checking similarities:
    - Chandler: max similarity = 0.1563
    - Joey: max similarity = -0.0508
    - Monica: max similarity = 0.0655
    - Phoebe: max similarity = 0.0623
    - Rachel: max similarity = 0.0665
    - Ross: max similarity = 0.7648
  all (9).jpg - Face #0: Ross (score=0.7648)
  
  Face #1 - Checking similarities:
    - Chandler: max similarity = 0.1004
    - Joey: max similarity = 0.0910
    - Monica: max similarity = 0.1000
    - Phoebe: max similarity = 0.7084
    - Rachel: max similarity = 0.1538
    - Ross: max similarity = 0.0386
  all (9).jpg - Face #1: Phoebe (score=0.7084)
  
  Copied data/photos\all (9).jpg to output\Ross\all (9).jpg
  Copied data/photos\all (9).jpg to output\Phoebe\all (9).jpg
```

### Understanding the Output

- **Face Detection**: Shows bounding boxes and embedding dimensions for each detected face
- **Similarity Scores**: Compares each face against all reference people (0.0 to 1.0 scale)
- **Recognition Results**: Shows the best match and confidence score
- **File Operations**: Copies photos to appropriate output folders

## ⚙️ Configuration

### Threshold Settings

The recognition threshold can be adjusted in `main.py`:
```python
face_processor = FaceProcessor(reference_db, threshold=0.5)  # Adjust this value
```

- **Higher threshold (0.7-0.9)**: More strict matching, fewer false positives
- **Lower threshold (0.3-0.5)**: More lenient matching, more matches but potential false positives

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

## 🔧 Troubleshooting

### Common Issues

1. **No faces detected**: Ensure reference images contain clear, front-facing faces
2. **Low accuracy**: Add more reference images for each person
3. **Memory issues**: Process fewer photos at once or reduce image resolution
4. **Library conflicts**: Use the provided virtual environment

### Performance Tips

- Use high-quality reference images
- Include multiple angles and lighting conditions in reference photos
- Ensure good lighting in photos to be analyzed
- Consider image resolution for processing speed vs. accuracy trade-offs

## 📈 Accuracy and Performance

The system achieves high accuracy through:
- **Unified InsightFace model**: Consistent detection and recognition
- **512-dimensional embeddings**: Rich feature representation
- **Cosine similarity matching**: Robust comparison method
- **Multiple reference images**: Improved recognition reliability

## 🤝 Contributing

This project follows clean code principles and SOLID design patterns. When contributing:
- Maintain the modular architecture
- Add appropriate documentation
- Follow the existing code style
- Test with various image types and conditions

## 📄 License

This project is for educational and research purposes. Please respect the original dataset license from Kaggle.


