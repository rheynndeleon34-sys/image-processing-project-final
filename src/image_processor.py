"""
Enhanced Image Processing Module
Contains 22 image processing techniques including advanced filters.
Author: [Your Name] - Image Processing Programmer
Date: [Current Date]
"""

import cv2
import numpy as np
from pathlib import Path
import os
import random
from .filters.advanced_filters import AdvancedFilters

class ImageProcessor:
    """
    Enhanced class implementing 22 image processing techniques.
    Includes basic, advanced, and artistic techniques.
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
        
        # Define all 22 techniques with display names
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
            'vignette': 'Vignette Effect',
            
            # New Techniques (16-22)
            'ascii_art': 'ASCII Art Effect',
            'vhs_effect': 'VHS Tape Effect',
            'pointillism': 'Pointillism Generator',
            'security_camera': 'Security Camera Effect',
            'film_burn': 'Film Burn Transition',
            'embroidery': 'Embroidery Pattern',
            'edge_detection': 'Enhanced Edge Detection'
        }
        
        print(f"Initialized Enhanced Image Processor")
        print(f"  Techniques available: {len(self.techniques)}")
        print(f"  Input directory: {self.input_dir}")
        print(f"  Output directory: {self.output_dir}")
    
    # ===========================================
    # 22 IMAGE PROCESSING TECHNIQUES
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
    # NEW TECHNIQUES (16-22)
    # ===========================================
    
    def technique_ascii_art(self, image):
        """
        Technique 16: ASCII Art Effect (text-based, retro computing)
        Creates a low-resolution effect that mimics ASCII art
        """
        # Resize to small dimensions for ASCII effect
        height, width = image.shape[:2]
        small_h, small_w = 80, 80
        resized = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to grayscale
        gray = self.technique_grayscale(resized)
        
        # Upscale back to original size with nearest neighbor for blocky effect
        ascii_effect = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Add contrast for better ASCII-like appearance
        ascii_effect = cv2.convertScaleAbs(ascii_effect, alpha=1.5, beta=30)
        
        # Convert back to 3 channels if needed
        if len(ascii_effect.shape) == 2:
            ascii_effect = cv2.cvtColor(ascii_effect, cv2.COLOR_GRAY2BGR)
        
        return ascii_effect
    
    def technique_vhs_effect(self, image):
        """
        Technique 17: VHS Tape Effect (analog video degradation)
        Simulates VHS tape artifacts: noise, color bleeding, scan lines
        """
        # Create VHS color distortion
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Add color shift (VHS color bleeding effect)
        hsv[:,:,0] = np.clip(hsv[:,:,0] + np.random.randint(-10, 10, hsv[:,:,0].shape), 0, 179)
        
        # Convert back to BGR
        vhs_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add noise (VHS tape noise)
        noise = np.random.normal(0, 25, vhs_color.shape).astype(np.uint8)
        vhs_noise = cv2.add(vhs_color, noise)
        
        # Add scan lines (VHS horizontal lines)
        height, width = vhs_noise.shape[:2]
        for i in range(0, height, 3):
            vhs_noise[i:i+1, :] = vhs_noise[i:i+1, :] * 0.7
        
        # Add slight blur for VHS softness
        vhs_blur = cv2.GaussianBlur(vhs_noise, (3, 3), 0)
        
        return vhs_blur
    
    def technique_pointillism(self, image):
        """
        Technique 18: Pointillism Generator (classic art technique)
        Creates a pointillism painting effect with colored dots
        """
        height, width = image.shape[:2]
        pointillism = image.copy()
        
        # Create a blank canvas
        canvas = np.zeros_like(image)
        
        # Determine dot size based on image dimensions
        dot_size = max(3, min(7, height // 100))
        
        # Sample points and draw colored circles
        num_points = (height * width) // 800  # Adjust density
        
        for _ in range(num_points):
            # Random position
            y = np.random.randint(0, height)
            x = np.random.randint(0, width)
            
            # Get color from original image
            color = image[y, x].tolist()
            
            # Draw circle with the sampled color
            cv2.circle(canvas, (x, y), dot_size, color, -1)
        
        # Blend with original for texture
        pointillism = cv2.addWeighted(canvas, 0.8, image, 0.2, 0)
        
        # Add slight blur to soften
        pointillism = cv2.GaussianBlur(pointillism, (3, 3), 0)
        
        return pointillism
    
    def technique_security_camera(self, image):
        """
        Technique 19: Security Camera Effect (modern surveillance aesthetic)
        Creates a low-quality security camera look with timestamp
        """
        # Convert to grayscale for security camera look
        gray = self.technique_grayscale(image)
        
        # Add noise (security camera interference)
        noise = np.random.normal(0, 15, gray.shape).astype(np.uint8)
        security = cv2.add(gray, noise)
        
        # Lower resolution effect
        height, width = security.shape
        small = cv2.resize(security, (width//2, height//2), interpolation=cv2.INTER_LINEAR)
        security = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Add timestamp overlay (simulated)
        timestamp = "23:59:45 01/01/2024"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(timestamp, font, 0.5, 1)[0]
        
        # Create a black rectangle for timestamp
        cv2.rectangle(security, (10, height - 30), (text_size[0] + 20, height - 10), 0, -1)
        
        # Add white timestamp text
        security = cv2.putText(security, timestamp, (15, height - 15), 
                              font, 0.5, 255, 1, cv2.LINE_AA)
        
        # Convert back to 3 channels if needed
        if len(security.shape) == 2:
            security = cv2.cvtColor(security, cv2.COLOR_GRAY2BGR)
        
        return security
    
    def technique_film_burn(self, image):
        """
        Technique 20: Film Burn Transition (cinematic drama)
        Creates a film burn effect with light leaks and color shifts
        """
        height, width = image.shape[:2]
        film_burn = image.copy()
        
        # Create light leak effect (yellow/orange overlay)
        light_leak = np.zeros_like(image)
        
        # Add gradient light leak
        for i in range(height):
            # Create orange-yellow gradient
            intensity = int(255 * (1 - abs(i - height/2) / (height/2)))
            light_leak[i, :] = [0, intensity//2, intensity]  # BGR format
        
        # Add random burn spots
        for _ in range(5):  # Number of burn spots
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            radius = np.random.randint(20, min(width, height)//4)
            
            # Create circular burn
            for y in range(max(0, center_y - radius), min(height, center_y + radius)):
                for x in range(max(0, center_x - radius), min(width, center_x + radius)):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < radius:
                        # Darken pixels near burn center
                        darkness = int(255 * (1 - dist/radius) * 0.7)
                        film_burn[y, x] = np.clip(film_burn[y, x] - darkness, 0, 255)
        
        # Blend with light leak
        film_burn = cv2.addWeighted(film_burn, 0.8, light_leak, 0.3, 0)
        
        # Add film grain
        grain = np.random.normal(0, 10, film_burn.shape).astype(np.uint8)
        film_burn = cv2.add(film_burn, grain)
        
        # Add vignette for dramatic effect
        vignette = self.technique_vignette(film_burn)
        
        return vignette
    
    def technique_embroidery(self, image):
        """
        Technique 21: Embroidery Pattern
        Creates an embroidery/stiching effect with pattern-like edges
        """
        # Edge detection for embroidery outline
        gray = self.technique_grayscale(image)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to make them thicker (like embroidery thread)
        kernel = np.ones((2, 2), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Create embroidery effect
        embroidery = image.copy()
        
        # Convert edges to color (red for embroidery look)
        colored_edges = np.zeros_like(image)
        colored_edges[thick_edges > 0] = [0, 0, 255]  # Red color in BGR
        
        # Reduce color palette for fabric-like appearance
        reduced_colors = cv2.convertScaleAbs(image, alpha=0.7, beta=30)
        
        # Combine reduced colors with embroidery edges
        embroidery = cv2.addWeighted(reduced_colors, 0.9, colored_edges, 0.4, 0)
        
        # Add texture (simulating fabric)
        texture = np.random.normal(0, 5, embroidery.shape).astype(np.uint8)
        embroidery = cv2.add(embroidery, texture)
        
        return embroidery
    
    def technique_edge_detection(self, image):
        """
        Technique 22: Enhanced Edge Detection
        More sophisticated edge detection with artistic presentation
        """
        # Convert to grayscale
        gray = self.technique_grayscale(image)
        
        # Apply multiple edge detection methods
        # 1. Sobel edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        
        # 2. Laplacian edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
        
        # 3. Canny edges
        canny = cv2.Canny(gray, 50, 150)
        
        # Combine edges with different weights
        combined = cv2.addWeighted(sobel, 0.4, laplacian, 0.3, 0)
        combined = cv2.addWeighted(combined, 0.7, canny, 0.3, 0)
        
        # Invert for black edges on white background
        edges_inverted = cv2.bitwise_not(combined)
        
        # Convert to 3 channels if needed
        if len(edges_inverted.shape) == 2:
            edges_inverted = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR)
        
        return edges_inverted
    
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
            'vignette': self.technique_vignette,
            'ascii_art': self.technique_ascii_art,
            'vhs_effect': self.technique_vhs_effect,
            'pointillism': self.technique_pointillism,
            'security_camera': self.technique_security_camera,
            'film_burn': self.technique_film_burn,
            'embroidery': self.technique_embroidery,
            'edge_detection': self.technique_edge_detection
        }
        
        if technique_name in technique_methods:
            return technique_methods[technique_name](image)
        else:
            print(f"  Warning: Unknown technique '{technique_name}'")
            return image
    
    def process_single_image(self, image_path):
        """
        Apply all 22 techniques to a single image.
        
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


# Example usage demonstration
if __name__ == "__main__":
    # Create an instance of the ImageProcessor
    processor = ImageProcessor(input_dir="images", output_dir="processed")
    
    # Process all images with all techniques
    success, total, stats = processor.process_all_images()
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Successfully processed: {success}/{total} images")
    print(f"Total files created: {stats['files_created']}")
    print(f"Techniques available: {stats['total_techniques']}")
    
    # Show available techniques
    print("\nAvailable techniques:")
    for i, (key, name) in enumerate(processor.techniques.items(), 1):
        print(f"{i:2d}. {name} ({key})")
    
    # Test a single technique on an example image
    print("\nTesting new techniques on sample image:")
    sample_image = np.ones((300, 300, 3), dtype=np.uint8) * 128
    
    # Test ASCII Art effect
    ascii_result = processor.technique_ascii_art(sample_image)
    print(f"ASCII Art effect created: {ascii_result.shape}")
    
    # Test VHS effect
    vhs_result = processor.technique_vhs_effect(sample_image)
    print(f"VHS effect created: {vhs_result.shape}")