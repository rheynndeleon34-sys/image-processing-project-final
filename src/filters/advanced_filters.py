import cv2
import numpy as np
from scipy import ndimage
import random

class AdvancedFilters:
    """
    Advanced image processing techniques for specialized effects.
    All methods are fixed to avoid size mismatch errors.
    """
    
    @staticmethod
    def oil_painting_effect(image, size=7):
        """
        Apply oil painting effect to image.
        
        Args:
            image: Input image
            size: Brush size (default: 7)
            
        Returns:
            Oil painting style image
        """
        # For faster processing, resize if image is too large
        if image.shape[0] > 800 or image.shape[1] > 800:
            scale = 0.5
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            small = image.copy()
            scale = 1.0
        
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small
        
        height, width = gray.shape[:2]
        
        if len(small.shape) == 3:
            output = np.zeros_like(small)
        else:
            output = np.zeros_like(gray)
        
        # Apply simplified oil painting effect
        for i in range(0, height, size):
            for j in range(0, width, size):
                i_end = min(i + size, height)
                j_end = min(j + size, width)
                
                window = gray[i:i_end, j:j_end]
                if window.size > 0:
                    # Find most frequent intensity in window
                    hist = np.bincount(window.flatten(), minlength=256)
                    max_intensity = np.argmax(hist)
                    
                    if len(small.shape) == 3:
                        for channel in range(3):
                            output[i:i_end, j:j_end, channel] = max_intensity
                    else:
                        output[i:i_end, j:j_end] = max_intensity
        
        # Resize back if needed
        if scale != 1.0:
            output = cv2.resize(output, (image.shape[1], image.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        
        return output.astype(np.uint8)
    
    @staticmethod
    def cartoon_effect(image):
        """
        Apply cartoon effect to image - FIXED VERSION.
        No more size mismatch errors.
        
        Args:
            image: Input image
            
        Returns:
            Cartoon-style image
        """
        # Make a copy
        img_copy = image.copy()
        
        # For color images
        if len(img_copy.shape) == 3:
            # 1. Apply bilateral filter for color smoothing
            # Reduce d value for faster processing if image is large
            if img_copy.shape[0] > 1000 or img_copy.shape[1] > 1000:
                color = cv2.bilateralFilter(img_copy, d=5, sigmaColor=50, sigmaSpace=50)
            else:
                color = cv2.bilateralFilter(img_copy, d=9, sigmaColor=75, sigmaSpace=75)
            
            # 2. Convert to grayscale for edge detection
            gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            
            # 3. Apply median blur
            gray = cv2.medianBlur(gray, 7)
            
            # 4. Detect edges using adaptive threshold
            edges = cv2.adaptiveThreshold(
                gray, 
                255, 
                cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 
                9, 
                2
            )
            
            # 5. Convert edges to 3-channel BGR
            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # 6. Combine using bitwise AND
            # DOUBLE CHECK sizes match
            if edges_color.shape == color.shape:
                cartoon = cv2.bitwise_and(color, edges_color)
            else:
                # Force resize if shapes don't match (safety check)
                print(f"  Warning: Resizing edges to match color shape")
                edges_color = cv2.resize(edges_color, (color.shape[1], color.shape[0]))
                cartoon = cv2.bitwise_and(color, edges_color)
            
            return cartoon
        
        else:
            # For grayscale images, simplified version
            gray = cv2.medianBlur(img_copy, 7)
            edges = cv2.adaptiveThreshold(
                gray, 
                255, 
                cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 
                9, 
                2
            )
            return edges
    
    @staticmethod
    def hdr_effect(image, alpha=1.5, beta=0.3):
        """
        Apply HDR (High Dynamic Range) effect.
        Simplified to avoid errors.
        
        Args:
            image: Input image
            alpha: Contrast enhancement factor
            beta: Saturation enhancement factor
            
        Returns:
            HDR-style image
        """
        # Convert to float32 for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Apply gamma correction
        img_gamma = np.power(img_float, 1.0/alpha)
        
        # For color images, enhance saturation in HSV space
        if len(image.shape) == 3:
            # Convert to HSV
            hsv = cv2.cvtColor((img_gamma * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Increase saturation
            s = cv2.add(s, int(50 * beta))
            s = np.clip(s, 0, 255)
            
            # Merge back
            hsv = cv2.merge([h, s, v])
            
            # Convert back to BGR
            img_hdr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            img_hdr = (img_gamma * 255).astype(np.uint8)
        
        # Apply CLAHE for local contrast enhancement
        if len(image.shape) == 3:
            lab = cv2.cvtColor(img_hdr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge([l, a, b])
            img_hdr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return img_hdr
    
    @staticmethod
    def watercolor_effect(image):
        """
        Apply watercolor painting effect.
        
        Args:
            image: Input image
            
        Returns:
            Watercolor-style image
        """
        # Apply bilateral filter for smoothness
        smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Apply median blur
        blurred = cv2.medianBlur(smoothed, 7)
        
        # For color images, enhance colors slightly
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Increase saturation for watercolor effect
            s = cv2.add(s, 20)
            s = np.clip(s, 0, 255)
            
            # Merge back
            hsv = cv2.merge([h, s, v])
            watercolor = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            watercolor = blurred
        
        return watercolor
    
    @staticmethod
    def vignette_effect(image, vignette_strength=0.7):
        """
        Apply vignette (darkened corners) effect.
        
        Args:
            image: Input image
            vignette_strength: Strength of vignette (0-1)
            
        Returns:
            Image with vignette effect
        """
        height, width = image.shape[:2]
        
        # Create vignette mask using Gaussian kernel
        kernel_x = cv2.getGaussianKernel(width, width/3)
        kernel_y = cv2.getGaussianKernel(height, height/3)
        kernel = kernel_y * kernel_x.T
        
        # Normalize the kernel
        mask = kernel / kernel.max()
        
        # Apply strength
        mask = 1 - (1 - mask) * vignette_strength
        
        # Reshape mask for color images
        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        # Apply vignette
        vignette_image = (image.astype(np.float32) * mask).astype(np.uint8)
        
        return vignette_image
    
    @staticmethod
    def texture_synthesis(image, patch_size=32):
        """
        Apply texture synthesis effect.
        
        Args:
            image: Input image
            patch_size: Size of texture patches
            
        Returns:
            Texturized image
        """
        # Simple texture synthesis by tiling
        height, width = image.shape[:2]
        
        # Create a 2x larger output
        if len(image.shape) == 3:
            output = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
        else:
            output = np.zeros((height * 2, width * 2), dtype=np.uint8)
        
        # Tile the image
        for i in range(0, output.shape[0], height):
            for j in range(0, output.shape[1], width):
                i_end = min(i + height, output.shape[0])
                j_end = min(j + width, output.shape[1])
                
                patch_height = i_end - i
                patch_width = j_end - j
                
                output[i:i_end, j:j_end] = image[:patch_height, :patch_width]
        
        return output
    
    @staticmethod
    def pop_art_effect(image, num_colors=4):
        """
        Apply pop art (Warhol-style) effect.
        
        Args:
            image: Input image
            num_colors: Number of colors for quantization
            
        Returns:
            Pop art style image
        """
        # Resize for faster processing
        small = cv2.resize(image, (200, 200))
        
        if len(small.shape) == 3:
            # Reshape to 2D array of pixels
            pixels = small.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            # Define criteria for k-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            
            # Apply k-means clustering
            _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to uint8
            centers = np.uint8(centers)
            
            # Map labels to centers
            quantized = centers[labels.flatten()]
            quantized = quantized.reshape(small.shape)
        else:
            # For grayscale, simple quantization
            quantized = ((small // (256 // num_colors)) * (256 // num_colors)).astype(np.uint8)
        
        # Apply posterization
        posterized = cv2.convertScaleAbs(quantized, alpha=1.2, beta=0)
        
        # Resize back to original size
        pop_art = cv2.resize(posterized, (image.shape[1], image.shape[0]))
        
        return pop_art