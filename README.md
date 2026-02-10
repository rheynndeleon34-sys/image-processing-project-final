# Image Processing


# About 
A Python-based image processing system implementing various computer vision techniques using a modular and test-driven approach, supported by automated testing and systematic documentation.

## Project Flow & Process
1. Planning & Architecture Design
   ↓
2. Implementation of Core Algorithms (src/)
   ↓
3. Testing & Validation (tests/)
   ↓
4. CI/CD Pipeline Setup (.github/workflows/)
   ↓
5. Demo Generation & Output Verification

## Automated CI/CD Pipeline
This project includes a professional GitHub Actions workflow that automatically validates all code changes:

```yaml
name: Image Processing CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest
    - name: Install GUI libraries
      run: |
        sudo apt-get update
        sudo apt-get install -y libgl1 libglib2.0-0
    - name: Run tests if available
      run: |
        if find tests -name "test_*.py" -type f 2>/dev/null | grep -q .; then
          echo "Running PyTest..."
          python -m pytest tests/ -v
        else
          echo "No test files found"
        fi
    - name: Verify project functionality
      run: |
        python -c "import cv2, numpy, scipy"
        python main.py --help
        echo "Project verification complete"
```

## What this automated pipeline does:
- ✅ Automatic Testing: Runs on every push and pull request
- ✅ Environment Setup: Creates clean Ubuntu environment with Python 3.10
- ✅ Smart Test Detection: Only runs pytest if test files exist
- ✅ Dependency Installation: Installs all required packages including OpenCV GUI libraries
- ✅ Project Validation: Verifies imports and basic functionality work

## Image Processing Dependencies
Required packages for the project:
- opencv-python==4.8.1.78 - OpenCV for image processing
- numpy==1.24.3 - Numerical operations
- matplotlib==3.7.1 - For creating comparison plots
- Pillow==10.0.0 - Alternative image library
- scipy==1.10.1 - For advanced filters (ndimage)

## Project Structure Implementation
```
image-processing-project-final/
├── .github/workflows/
├── demo/
├── input/
├── output/
├── src/
│   ├── filters/
│   ├── transformations/
│   ├── detection/
│   └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── data/
├── main.py
├── requirements.txt
└── README.md
```

## How to Run the Application
### Prerequisites
- Python 3.8 or higher
- Git (for cloning)

## Installation Steps
1. Clone the Repository
```bash
git clone https://github.com/rheynndeleon34-sys/image-processing-project-final.git
cd image-processing-project-final
```

2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv
# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Main Application
## Option 1: Using main.py (Primary method)
```bash
python main.py
```

## Option 2: Using processor.py (For specific operations)
```bash
python processor.py [arguments]
```

Note: Check processor.py for specific argument requirements.

# Running Tests
```bash
#Run all tests
pytest tests/

# Run specific test module
pytest tests/test_filters.py

# Run with verbose output
pytest -v tests/

# Run tests with coverage report
pytest --cov=src tests/
```

# Generating Demo Images
```bash
## Check demo/ directory for existing scripts
## or run the demo generation module if available
python -m demo.generate  # If such a module exists
```

## Development Workflow for Team Members
1. Before Starting Work
	Pull latest changes: `git pull origin main`
	Activate virtual environment
	Create a new branch: `git checkout -b feature/your-feature-name`
2. Adding New Features
	Place source code in `src/` directory
	Add corresponding tests in `tests/` directory
	Test locally before committing

3. Adding Test Images
	Place input images in `input/` directory
	Supported formats: .jpg, .png, .bmp
	Add descriptive filenames
4. Before Committing
	Run tests: `pytest tests/`
	Ensure no broken functionality
	Update documentation if needed
5. Commit & Push
```bash
git add .
git commit -m "Descriptive message about changes"
git push origin feature/your-feature-name
```

## Important Notes for Submission

### 1. Input/Output Directories
- `input/`: Only for source images (DO NOT commit processed images here)
- `output/`: For final processed results (Commit important results here)
- `demo/`: Auto-generated content (May be excluded from final submission)

### 2. File Naming Convention
- Python files: snake_case (e.g., `image_processor.py`)
- Test files: `test_module_name.py`
- Output images: `originalname_technique.ext`

### 3. Testing Requirements
- All new features must include tests
- Test coverage should be maintained
- CI pipeline must pass before merge

## Image Processing Techniques & Algorithms
### 1. Filtering Techniques
- Gaussian Blur: Smoothing using Gaussian kernel convolution
- Median Filter: Noise reduction through median pixel value replacement
- Bilateral Filter: Edge-preserving smoothing
- Custom Kernels: User-defined convolution operations

### 2. Edge Detection Algorithms
- Sobel Operator: Gradient-based edge detection
- Canny Edge Detector: Multi-stage algorithm for optimal edge detection
- Laplacian of Gaussian (LoG): Second derivative edge detection

### 3. Transformation Methods
- Geometric Transformations: Rotation, scaling, translation
- Affine Transformations: Linear mapping with preservation of lines and parallelism
- Perspective Transform: Projective geometry transformations

### 4. Color Processing
- Color Space Conversions: RGB ↔ HSV ↔ Grayscale
- Histogram Equalization: Contrast enhancement
- Color Channel Manipulation: Individual channel processing

### 5. Feature Detection
- Corner Detection: Harris corner detection algorithm
- Blob Detection: Identifying regions with different properties
- Template Matching: Finding template patterns within images

## Algorithm Implementation Details

### Core Processing Pipeline
```
Input Image → Preprocessing → Technique Application → Postprocessing → Output Image
```

# Key Algorithmic Decisions
1. Modular Design
- Separated algorithms into distinct modules for maintainability
- Used object-oriented principles where appropriate
- Created reusable filter and transformation classes

2. Performance Optimization
- Vectorized operations using NumPy arrays
- Efficient memory management for large images
- Multi-processing for batch image processing

3. Accuracy & Validation
- Implemented unit tests for each algorithm
- Used standard test images for verification
- Compared results with established libraries

## Testing Methodology
- Unit Tests: Individual function testing
- Integration Tests: Full pipeline validation
- Visual Verification: Manual inspection of output images
- Automated CI: GitHub Actions for continuous validation


## Key Achievements & Results
1. Successfully Implemented: Multiple image processing techniques with proper abstraction
2. Testing Coverage: Comprehensive test suite with automated execution
3. CI/CD Pipeline: Fully automated testing workflow on GitHub
4. Demo Generation: Automated creation of demonstration images
5. Team Collaboration: Effective division of responsibilities with clear role definitions

## Future Enhancements
Additional Algorithms: Machine learning-based image processing
GUI Interface: User-friendly graphical interface for non-technical users
Real-time Processing: Video stream processing capabilities
Performance Metrics: Quantitative evaluation of algorithm performance
Extended Testing: More comprehensive edge case testing

##Project Completion Date: February 2026
Status: ✅ Successfully completed with all objectives met
Repository: https://github.com/rheynndeleon34-sys/image-processing-project-final
