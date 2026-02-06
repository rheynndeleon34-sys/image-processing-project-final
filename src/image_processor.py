import cv2
import numpy as np
from pathlib import Path
import os
from .filters.advanced_filters import AdvancedFilters

class ImageProcessor:
    """
    Enhanced class implementing 15 image processing techniques.
    Includes basic and advanced techniques.
    """
    
    def __init__(self, input_dir="input", output_dir="output"):
        """
        Initialize the enhanced image processor.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save processed images
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Supported image extensions
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        # Initialize advanced filters
        self.advanced = AdvancedFilters()
        
        # Define all 15 techniques with display names
        self.techniques = {
            # Basic Techniques (1-10)
            'grayscale': 'Grayscale',
            'canny_edge': 'Canny Edge Detection',
            'color_invert': 'Color Inversion',
            'gaussian_blur': 'Gaussian Blur',
            'sepia_tone': 'Sepia Tone',
            'pencil_sketch': 'Pencil Sketch',
            'sharpen': 'Image Sharpening',
            'brightness_contrast': 'Brightness & Contrast',
            'binary_threshold': 'Binary Threshold',
            'emboss': 'Emboss Effect',
            
            # Advanced Techniques (11-15)
            'oil_painting': 'Oil Painting',
            'cartoon': 'Cartoon Effect',
            'hdr': 'HDR Effect',
            'watercolor': 'Watercolor',
            'vignette': 'Vignette Effect'
        }
        
        print(f"Initialized Enhanced Image Processor")
        print(f"  Techniques available: {len(self.techniques)}")
        print(f"  Input directory: {self.input_dir}")
        print(f"  Output directory: {self.output_dir}")
    
    # ===========================================
    # 15 IMAGE PROCESSING TECHNIQUES
    # ===========================================
    
    # Basic Techniques (1-10) - Keep existing implementations
    def technique_grayscale(self, image):
        """Technique 1: Convert to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def technique_canny_edge(self, image):
        """Technique 2: Canny edge detection"""
        gray = self.technique_grayscale(image)
        return cv2.Canny(gray, 30, 100)
    
    def technique_color_invert(self, image):
        """Technique 3: Color inversion"""
        return cv2.bitwise_not(image)
    
    def technique_gaussian_blur(self, image, kernel_size=15):
        """Technique 4: Gaussian blur"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def technique_sepia_tone(self, image):
        """Technique 5: Sepia tone filter"""
        sepia_filter = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia = cv2.transform(image, sepia_filter)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    
    def technique_pencil_sketch(self, image):
        """Technique 6: Pencil sketch"""
        gray = self.technique_grayscale(image)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        inverted_blurred = cv2.bitwise_not(blurred)
        return cv2.divide(gray, inverted_blurred, scale=256.0)
    
    def technique_sharpen(self, image):
        """Technique 7: Image sharpening"""
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, sharpen_kernel)
    
    def technique_brightness_contrast(self, image, alpha=1.3, beta=40):
        """Technique 8: Brightness/contrast adjustment"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def technique_binary_threshold(self, image, threshold=127):
        """Technique 9: Binary threshold"""
        gray = self.technique_grayscale(image)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    def technique_emboss(self, image):
        """Technique 10: Emboss effect"""
        emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        embossed = cv2.filter2D(image, -1, emboss_kernel)
        return cv2.addWeighted(embossed, 0.9, 128, 0.1, 0)
    
    # Advanced Techniques (11-15)
    def technique_oil_painting(self, image):
        """Technique 11: Oil painting effect"""
        return self.advanced.oil_painting_effect(image)
    
    def technique_cartoon(self, image):
        """Technique 12: Cartoon effect"""
        return self.advanced.cartoon_effect(image)
    
    def technique_hdr(self, image):
        """Technique 13: HDR effect"""
        return self.advanced.hdr_effect(image)
    
    def technique_watercolor(self, image):
        """Technique 14: Watercolor effect"""
        return self.advanced.watercolor_effect(image)
    
    def technique_vignette(self, image):
        """Technique 15: Vignette effect"""
        return self.advanced.vignette_effect(image, vignette_strength=0.7)
    
    # ===========================================
    # IMAGE PROCESSING PIPELINE
    # ===========================================
    
    def load_image(self, image_path):
        """Load an image from file path"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  Warning: Could not load {image_path}")
            return image
        except Exception as e:
            print(f"  Error loading {image_path}: {e}")
            return None
    
    def save_image(self, image, output_path):
        """Save image to file"""
        try:
            if len(image.shape) == 2:
                image_to_save = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_to_save = image
            cv2.imwrite(str(output_path), image_to_save)
        except Exception as e:
            print(f"  Error saving {output_path}: {e}")
    
    def get_all_image_paths(self):
        """Get all image file paths from input directory"""
        image_paths = []
        
        if not self.input_dir.exists():
            print(f"Error: Input directory '{self.input_dir}' does not exist!")
            return image_paths
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix.lower() in self.supported_extensions:
                    image_paths.append(file_path)
        
        return image_paths
    
    def apply_technique(self, technique_name, image):
        """
        Apply a specific technique to an image.
        
        Args:
            technique_name (str): Name of the technique
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Processed image
        """
        # Map technique name to method
        technique_methods = {
            'grayscale': self.technique_grayscale,
            'canny_edge': self.technique_canny_edge,
            'color_invert': self.technique_color_invert,
            'gaussian_blur': self.technique_gaussian_blur,
            'sepia_tone': self.technique_sepia_tone,
            'pencil_sketch': self.technique_pencil_sketch,
            'sharpen': self.technique_sharpen,
            'brightness_contrast': self.technique_brightness_contrast,
            'binary_threshold': self.technique_binary_threshold,
            'emboss': self.technique_emboss,
            'oil_painting': self.technique_oil_painting,
            'cartoon': self.technique_cartoon,
            'hdr': self.technique_hdr,
            'watercolor': self.technique_watercolor,
            'vignette': self.technique_vignette
        }
        
        if technique_name in technique_methods:
            return technique_methods[technique_name](image)
        else:
            print(f"  Warning: Unknown technique '{technique_name}'")
            return image
    
    def process_single_image(self, image_path):
        """
        Apply all 15 techniques to a single image.
        
        Args:
            image_path (Path): Path to input image
            
        Returns:
            dict: Dictionary of processed images
        """
        print(f"  Processing: {image_path.name}")
        
        # Load the image
        original_image = self.load_image(image_path)
        if original_image is None:
            return None
        
        # Apply all techniques
        processed_images = {'original': original_image}
        
        for technique_name in self.techniques.keys():
            try:
                processed_image = self.apply_technique(technique_name, original_image.copy())
                processed_images[technique_name] = processed_image
            except Exception as e:
                print(f"  Error applying {technique_name}: {e}")
                processed_images[technique_name] = original_image  # Fallback
        
        return processed_images
    
    def process_all_images(self, techniques_to_apply=None):
        """
        Process all images in the input directory.
        
        Args:
            techniques_to_apply (list): Optional list of specific techniques to apply
            
        Returns:
            tuple: (success_count, total_count, stats)
        """
        # Get all image files
        image_paths = self.get_all_image_paths()
        total_images = len(image_paths)
        
        if total_images == 0:
            print("No images found in input directory!")
            print(f"Supported formats: {', '.join(self.supported_extensions)}")
            return 0, 0, {}
        
        print(f"Found {total_images} image(s) to process")
        print(f"Applying {len(self.techniques)} techniques per image")
        print("-" * 60)
        
        success_count = 0
        stats = {
            'total_techniques': len(self.techniques),
            'images_processed': 0,
            'files_created': 0
        }
        
        # Determine which techniques to apply
        if techniques_to_apply is None:
            techniques_to_apply = list(self.techniques.keys())
        
        # Process each image
        for image_path in image_paths:
            # Process the image
            processed_images = self.process_single_image(image_path)
            
            if processed_images is None:
                continue
            
            # Save all processed versions
            base_name = image_path.stem
            
            for technique_name, processed_image in processed_images.items():
                # Only save if technique is in our list
                if technique_name in techniques_to_apply or technique_name == 'original':
                    output_filename = f"{base_name}_{technique_name}.png"
                    output_path = self.output_dir / output_filename
                    
                    self.save_image(processed_image, output_path)
                    stats['files_created'] += 1
            
            success_count += 1
            stats['images_processed'] += 1
            
            print(f"  ✓ Created {len(techniques_to_apply) + 1} versions of {image_path.name}")
        
        return success_count, total_images, stats