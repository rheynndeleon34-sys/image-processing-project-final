"""
Advanced Image Processing Module
Contains 25 image processing techniques including movie poster and album cover effects.
Author: [Your Name] - Image Processing Programmer
Date: [Current Date]
"""

import cv2
import numpy as np
from pathlib import Path
import os
import random
import math
from scipy import fftpack
import textwrap

class ImageProcessor:
    """
    Advanced class implementing 25 image processing techniques.
    """
    
    def __init__(self, input_dir="input", output_dir="output"):
        """
        Initialize the image processor.
        
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
        
        # Define all 25 techniques with display names
        self.techniques = {
            # Basic Techniques (1-8)
            'canny_edge': 'Canny Edge Detection',
            'anime_style': 'Anime Animation Style',
            'sepia_tone': 'Sepia Tone',
            'pencil_sketch': 'Pencil Sketch',
            'sharpen': 'Image Sharpening',
            'edge_detection': 'Enhanced Edge Detection',
            'binary_threshold': 'Binary Threshold',
            'emboss': 'Emboss Effect',
            
            # Advanced Techniques (9-13)
            'oil_painting': 'Oil Painting',
            'cartoon': 'Cartoon Effect',
            'hdr': 'HDR Effect',
            'watercolor': 'Watercolor',
            'vignette': 'Vignette Effect',
            
            # Artistic Effects (14-20)
            'movie_poster': 'Movie Poster Effect',
            'album_cover': 'Album Cover Effect',
            'vhs_effect': 'VHS Tape Effect',
            'pointillism': 'Pointillism Generator',
            'security_camera': 'Security Camera Effect',
            'film_burn': 'Film Burn Transition',
            'embroidery': 'Embroidery Pattern',
            
            # Computer Vision Techniques (21-25)
            'image_stitching': 'Panorama Stitching',
            'background_subtraction': 'Background Subtraction',
            'image_compression': 'Image Compression',
            'style_transfer': 'Style Transfer',
            'optical_flow': 'Optical Flow'
        }
        
        print(f"Initialized Advanced Image Processor")
        print(f"  Techniques available: {len(self.techniques)}")
        print(f"  Input directory: {self.input_dir}")
        print(f"  Output directory: {self.output_dir}")
    
    # ===========================================
    # 25 IMAGE PROCESSING TECHNIQUES
    # ===========================================
    
    # Basic Techniques (1-8)
    def technique_canny_edge(self, image):
        """Technique 1: Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Canny(gray, 30, 100)
    
    def technique_anime_style(self, image):
        """Technique 2: Anime Animation Style - Transform into anime/cartoon characters"""
        height, width = image.shape[:2]
        
        # Step 1: Reduce colors using bilateral filtering and posterization
        anime = cv2.bilateralFilter(image, 9, 100, 100)
        
        # Step 2: Apply strong edge detection for outline
        gray = cv2.cvtColor(anime, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 20, 60)
        
        # Dilate edges to make them more prominent
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Step 3: Reduce color palette (posterization)
        # Apply k-means to reduce to 8 main colors
        data = anime.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        result = centers[labels.flatten()]
        anime = result.reshape(image.shape)
        
        # Step 4: Enhance and saturate colors
        hsv = cv2.cvtColor(anime, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.6)  # Boost saturation
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        anime = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Step 5: Apply black outlines over the image
        # Create a version with black edges
        edges_inverted = cv2.bitwise_not(edges)
        edges_3ch = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR)
        
        # Blend the image with black edges
        anime = cv2.addWeighted(anime, 0.9, edges_3ch, 0.1, 0)
        
        # Step 6: Add subtle anime shading effect
        # Create shadow areas
        shading = cv2.GaussianBlur(gray, (25, 25), 0)
        shading = cv2.normalize(shading, None, 0, 100, cv2.NORM_MINMAX).astype(np.uint8)
        shading_3ch = cv2.cvtColor(shading, cv2.COLOR_GRAY2BGR)
        
        # Darken based on shading
        anime = cv2.addWeighted(anime, 1.0, shading_3ch, -0.05, 0)
        anime = np.clip(anime, 0, 255).astype(np.uint8)
        
        # Step 7: Add highlights (bright areas)
        # Find bright regions for highlight effect
        bright_mask = cv2.inRange(gray, 180, 255)
        bright_3ch = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR)
        anime = cv2.addWeighted(anime, 1.0, bright_3ch, 0.15, 0)
        
        return np.clip(anime, 0, 255).astype(np.uint8)
    
    def technique_sepia_tone(self, image):
        """Technique 3: Sepia tone filter"""
        sepia_filter = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        sepia = cv2.transform(image, sepia_filter)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    
    def technique_pencil_sketch(self, image):
        """Technique 4: Pencil sketch"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        inverted_blurred = cv2.bitwise_not(blurred)
        return cv2.divide(gray, inverted_blurred, scale=256.0)
    
    def technique_sharpen(self, image):
        """Technique 5: Image sharpening"""
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, sharpen_kernel)
    
    def technique_edge_detection(self, image):
        """Technique 6: Enhanced Edge Detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
        canny = cv2.Canny(gray, 50, 150)
        combined = cv2.addWeighted(sobel, 0.4, laplacian, 0.3, 0)
        combined = cv2.addWeighted(combined, 0.7, canny, 0.3, 0)
        edges_inverted = cv2.bitwise_not(combined)
        return cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR)
    
    def technique_binary_threshold(self, image, threshold=127):
        """Technique 7: Binary threshold"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary
    
    def technique_emboss(self, image):
        """Technique 8: Emboss effect"""
        emboss_kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        embossed = cv2.filter2D(image, -1, emboss_kernel)
        return cv2.addWeighted(embossed, 0.9, 128, 0.1, 0)
    
    # Advanced Techniques (9-13)
    def technique_oil_painting(self, image):
        """Technique 9: Oil painting effect"""
        # Apply strong bilateral filtering for smooth edges
        oil = cv2.bilateralFilter(image, 11, 100, 100)
        
        # Reduce colors for painted effect using k-means clustering
        data = oil.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 12, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        result = centers[labels.flatten()]
        oil = result.reshape(image.shape)
        
        # Apply median blur for painterly effect
        oil = cv2.medianBlur(oil, 7)
        
        # Add subtle edge enhancement
        edges = cv2.Canny(cv2.cvtColor(oil, cv2.COLOR_BGR2GRAY), 30, 100)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        
        # Blend original with processed for final oil effect
        result = cv2.addWeighted(oil, 0.85, image, 0.15, 0)
        
        return result
    
    def technique_cartoon(self, image):
        """Technique 10: Cartoon effect"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    
    def technique_hdr(self, image):
        """Technique 11: HDR effect"""
        hdr = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
        return cv2.convertScaleAbs(hdr, alpha=1.2, beta=20)
    
    def technique_watercolor(self, image):
        """Technique 12: Watercolor effect"""
        try:
            watercolor = cv2.stylization(image, sigma_s=60, sigma_r=0.6)
            return watercolor
        except:
            # Fallback
            bilateral = cv2.bilateralFilter(image, 15, 100, 100)
            return cv2.addWeighted(image, 0.3, bilateral, 0.7, 0)
    
    def technique_vignette(self, image, vignette_strength=0.7):
        """Technique 13: Vignette effect"""
        rows, cols = image.shape[:2]
        
        kernel_x = cv2.getGaussianKernel(cols, cols/vignette_strength)
        kernel_y = cv2.getGaussianKernel(rows, rows/vignette_strength)
        kernel = kernel_y * kernel_x.T
        
        mask = 255 * kernel / np.linalg.norm(kernel)
        mask = mask.reshape(rows, cols)
        
        vignette = np.copy(image)
        for i in range(3):
            vignette[:,:,i] = vignette[:,:,i] * mask
        
        return vignette.astype(np.uint8)
    
    # ===========================================
    # NEW ARTISTIC EFFECTS (14-15)
    # ===========================================
    
    def technique_movie_poster(self, image):
        """
        Technique 14: Movie Poster Effect
        Transforms image into a cinematic movie poster with dramatic effects
        """
        height, width = image.shape[:2]
        
        # Start with the original image - keep it sharp
        poster = image.copy()
        
        # Apply strong contrast enhancement for cinematic look
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        lab = cv2.cvtColor(poster, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        poster = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply dramatic color grading
        # Create teal and orange color grading
        hsv = cv2.cvtColor(poster, cv2.COLOR_BGR2HSV)
        
        # Boost saturation significantly for dramatic effect
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.6)
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        
        # Slightly increase value for brighter appearance
        hsv[:,:,2] = cv2.multiply(hsv[:,:,2], 1.1)
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
        
        poster = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add dramatic lighting effect (simulated)
        # Create a highlight effect on upper portion
        rows, cols = poster.shape[:2]
        light_effect = np.zeros_like(poster, dtype=np.float32)
        
        # Bright spotlight effect
        for y in range(rows):
            for x in range(cols):
                # Create radial gradient from top-left
                dist_norm = np.sqrt((x - cols * 0.3)**2 + (y - rows * 0.2)**2) / (rows * 0.5)
                brightness = max(0, 1 - dist_norm) * 0.3
                light_effect[y, x] = brightness
        
        poster_float = poster.astype(np.float32)
        poster_float[:,:,0] += light_effect[:,:,0] * 60
        poster_float[:,:,1] += light_effect[:,:,0] * 40
        poster_float[:,:,2] += light_effect[:,:,0] * 80  # More blue in highlights
        poster = np.clip(poster_float, 0, 255).astype(np.uint8)
        
        # Add subtle film grain for texture (not blur)
        grain = np.random.normal(0, 4, poster.shape).astype(np.int16)
        poster = np.clip(poster.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        
        # Add thick black borders for movie poster framing
        border_thickness = height // 12
        poster_framed = poster.copy()
        poster_framed[:border_thickness, :] = (poster_framed[:border_thickness, :] * 0.1).astype(np.uint8)
        poster_framed[-border_thickness:, :] = (poster_framed[-border_thickness:, :] * 0.1).astype(np.uint8)
        poster = poster_framed
        
        # Add decorative border frame
        cv2.rectangle(poster, (15, 15), (width-15, height-15), (200, 150, 50), 3)
        
        # Add movie title with large bold text
        titles = ["THE MOMENT", "AWAKENING", "BEYOND", "LEGACY", "RECKONING", "ASCEND", "REVELATION"]
        title = random.choice(titles)
        title_font = cv2.FONT_HERSHEY_TRIPLEX
        title_scale = max(width / 500, 2.0)
        
        (title_width, title_height), _ = cv2.getTextSize(title, title_font, title_scale, 4)
        title_x = (width - title_width) // 2
        title_y = height // 2
        
        # Draw title with multiple layers for dramatic effect
        # Shadow layer (black)
        cv2.putText(poster, title, (title_x+4, title_y+4), 
                   title_font, title_scale, (0, 0, 0), 5, cv2.LINE_AA)
        # Outline layer (gold)
        cv2.putText(poster, title, (title_x, title_y), 
                   title_font, title_scale, (0, 215, 255), 4, cv2.LINE_AA)
        # Main layer (bright white)
        cv2.putText(poster, title, (title_x, title_y), 
                   title_font, title_scale, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add dramatic tagline
        taglines = ["PREPARE FOR IMPACT", "REALITY AWAITS", "EVERYTHING CHANGES", "THE TRUTH REVEALED", "DESTINY CALLS"]
        tagline = random.choice(taglines)
        tag_font = cv2.FONT_HERSHEY_TRIPLEX
        tag_scale = width / 900
        
        (tag_width, tag_height), _ = cv2.getTextSize(tagline, tag_font, tag_scale, 2)
        tag_x = (width - tag_width) // 2
        tag_y = title_y + int(title_height * 2) + 30
        
        # Draw tagline with shadow
        cv2.putText(poster, tagline, (tag_x+2, tag_y+2), 
                   tag_font, tag_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(poster, tagline, (tag_x, tag_y), 
                   tag_font, tag_scale, (200, 220, 255), 2, cv2.LINE_AA)
        
        # Add release info prominently at bottom
        release = "COMING SOON"
        release_font = cv2.FONT_HERSHEY_TRIPLEX
        release_scale = width / 1000
        
        (release_width, _), _ = cv2.getTextSize(release, release_font, release_scale, 3)
        release_x = (width - release_width) // 2
        
        cv2.putText(poster, release, (release_x+2, height - 50), 
                   release_font, release_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(poster, release, (release_x, height - 50), 
                   release_font, release_scale, (255, 200, 100), 2, cv2.LINE_AA)
        
        # Add rating box (top right)
        ratings = ["PG-13", "R", "PG", "NC-17"]
        rating = random.choice(ratings)
        cv2.rectangle(poster, (width - 100, 30), (width - 20, 80), (0, 0, 0), 2)
        cv2.putText(poster, rating, (width - 90, 70), 
                   cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add decorative stars/ornaments
        star_y = title_y - 60
        for i in range(5):
            star_x = width // 2 - 100 + i * 50
            cv2.putText(poster, "★", (star_x, star_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 2, cv2.LINE_AA)
        
        return poster
    
    def technique_album_cover(self, image):
        """
        Technique 15: Album Cover Effect
        Transforms image into a music album cover with artistic elements
        """
        height, width = image.shape[:2]
        
        # Create album cover base
        album = image.copy()
        
        # Apply artistic filter - try multiple effects
        if random.random() > 0.5:
            # Option 1: High contrast pop art style
            hsv = cv2.cvtColor(album, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.8)  # Boost saturation
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            album = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Posterize effect (reduce colors)
            album = (album // 64) * 64 + 32
        else:
            # Option 2: Moody, desaturated look
            album = cv2.cvtColor(album, cv2.COLOR_BGR2GRAY)
            album = cv2.cvtColor(album, cv2.COLOR_GRAY2BGR)
            album = cv2.convertScaleAbs(album, alpha=0.8, beta=30)
        
        # Add vinyl record effect (circular gradient)
        center_x, center_y = width // 2, height // 2
        radius = min(center_x, center_y) // 2
        
        # Create circular gradient mask
        y_coords, x_coords = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Apply circular vignette
        mask = np.clip(1 - dist_from_center / (radius * 1.5), 0, 1)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        album = (album * mask).astype(np.uint8)
        
        # Add album title (random music-related titles)
        album_titles = [
            "ECHOES OF TIME",
            "SILENT WHISPERS", 
            "URBAN LEGENDS",
            "DIGITAL DREAMS",
            "MIDNIGHT SESSIONS",
            "SOLITUDE",
            "THE AWAKENING",
            "LOST FREQUENCIES"
        ]
        album_title = random.choice(album_titles)
        
        title_font = cv2.FONT_HERSHEY_DUPLEX
        title_scale = width / 700
        
        # Wrap text if too long
        if len(album_title) > 15:
            wrapped = textwrap.wrap(album_title, width=12)
            for i, line in enumerate(wrapped):
                (text_width, text_height), _ = cv2.getTextSize(
                    line, title_font, title_scale, 2
                )
                text_x = (width - text_width) // 2
                text_y = 80 + i * (text_height + 10)
                
                cv2.putText(album, line, (text_x, text_y), 
                           title_font, title_scale, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            (text_width, text_height), _ = cv2.getTextSize(
                album_title, title_font, title_scale, 2
            )
            text_x = (width - text_width) // 2
            text_y = 80
            
            cv2.putText(album, album_title, (text_x, text_y), 
                       title_font, title_scale, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add artist name
        artists = ["THE VISUALS", "PIXEL PERFECT", "AI COLLECTIVE", "DIGITAL SOUND"]
        artist = random.choice(artists)
        
        artist_font = cv2.FONT_HERSHEY_SIMPLEX
        artist_scale = width / 1000
        
        (artist_width, _), _ = cv2.getTextSize(artist, artist_font, artist_scale, 1)
        artist_x = (width - artist_width) // 2
        
        cv2.putText(album, artist, (artist_x, height - 50), 
                   artist_font, artist_scale, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Add music genre label
        genres = ["ALTERNATIVE", "ELECTRONIC", "EXPERIMENTAL", "AMBIENT"]
        genre = random.choice(genres)
        
        cv2.putText(album, genre, (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Add barcode (simulated)
        bar_height = 20
        bar_width = 2
        bar_start_x = width - 150
        bar_start_y = height - 40
        
        for i in range(50):
            bar_x = bar_start_x + i * bar_width
            bar_h = random.randint(5, bar_height)
            cv2.rectangle(album, 
                         (bar_x, bar_start_y + (bar_height - bar_h)),
                         (bar_x + bar_width - 1, bar_start_y + bar_height),
                         (0, 0, 0), -1)
        
        # Add record label logo
        labels = ["DIGITAL RECORDS", "PIXEL WAX", "BYTE BEATS"]
        label = random.choice(labels)
        
        cv2.putText(album, "© " + label, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        
        return album
    
    # ===========================================
    # EXISTING ARTISTIC EFFECTS (16-20)
    # ===========================================
    
    def technique_vhs_effect(self, image):
        """Technique 16: VHS Tape Effect"""
        vhs = image.copy().astype(np.float32)
        height, width = vhs.shape[:2]
        
        # Add chromatic aberration (color shift) - typical VHS artifact
        channels = list(cv2.split(vhs))  # Convert tuple to list for modification
        shift_amount = np.random.randint(2, 5)
        
        # Shift color channels slightly
        channels[0] = np.roll(channels[0], shift_amount, axis=1)  # Blue channel shift
        channels[2] = np.roll(channels[2], -shift_amount, axis=1)  # Red channel shift
        vhs = cv2.merge(channels)
        
        # Add severe color noise/distortion
        hsv = cv2.cvtColor(vhs.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] += np.random.randint(-20, 20, hsv[:,:,0].shape)  # Hue shift
        hsv[:,:,1] *= np.random.uniform(0.5, 1.5, hsv[:,:,1].shape)  # Saturation variation
        hsv = np.clip(hsv, [0, 0, 0], [179, 255, 255])
        vhs = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        
        # Add heavy analog noise
        noise = np.random.normal(0, 30, vhs.shape).astype(np.float32)
        vhs = cv2.add(vhs, noise)
        
        # Add scan lines (horizontal artifacts)
        for i in range(0, height, 2):
            line_intensity = np.random.uniform(0.4, 0.8)
            vhs[i:i+1, :] *= line_intensity
        
        # Add tracking noise (vertical glitches)
        for _ in range(3):
            glitch_y = np.random.randint(0, height)
            glitch_height = np.random.randint(5, 20)
            glitch_x_shift = np.random.randint(-10, 10)
            
            if glitch_y + glitch_height < height:
                vhs[glitch_y:glitch_y+glitch_height, :] = np.roll(
                    vhs[glitch_y:glitch_y+glitch_height, :], glitch_x_shift, axis=1
                )
        
        # Add horizontal instability
        for i in range(height):
            shift = np.random.randint(-3, 3)
            vhs[i, :] = np.roll(vhs[i, :], shift, axis=0)
        
        vhs = np.clip(vhs, 0, 255).astype(np.uint8)
        
        # Apply slight motion blur for that tape motion feel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vhs = cv2.filter2D(vhs, -1, kernel)
        
        return vhs
    
    def technique_pointillism(self, image):
        """Technique 17: Pointillism Generator"""
        height, width = image.shape[:2]
        
        # Increase number of points for more realistic pointillism
        num_points = (height * width) // 200  # Much more dots than before
        dot_sizes = [2, 3, 4, 5, 6, 7, 8]  # Vary dot sizes for more interesting effect
        
        pointillism = image.copy()
        
        # Apply multiple layers of dots with varying opacity
        for layer in range(2):
            canvas = np.zeros_like(image, dtype=np.float32)
            points_per_layer = num_points // 2
            
            for _ in range(points_per_layer):
                y = np.random.randint(0, height)
                x = np.random.randint(0, width)
                dot_size = random.choice(dot_sizes)
                
                # Get color with slight variation
                color = image[y, x].astype(np.float32)
                # Add slight color variation for artistic effect
                variation = np.random.uniform(0.85, 1.15, 3)
                color = np.clip(color * variation, 0, 255)
                
                # Draw dot with anti-aliasing
                cv2.circle(canvas, (x, y), dot_size, color, -1, cv2.LINE_AA)
            
            # Blend layer
            alpha = 0.5 if layer == 0 else 0.3
            pointillism = cv2.addWeighted(pointillism.astype(np.float32), 1.0, canvas, alpha, 0)
        
        pointillism = np.clip(pointillism, 0, 255).astype(np.uint8)
        
        # Add very subtle texture to simulate canvas
        canvas_texture = np.random.normal(128, 3, pointillism.shape).astype(np.uint8)
        pointillism = cv2.addWeighted(pointillism, 0.95, canvas_texture, 0.05, 0)
        
        return pointillism
    
    def technique_security_camera(self, image):
        """Technique 18: Security Camera Effect"""
        height, width = image.shape[:2]
        
        # Convert to grayscale for CCTV look
        security = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Reduce resolution (typical of old security cameras)
        reduced = cv2.resize(security, (width//3, height//3), interpolation=cv2.INTER_LINEAR)
        security = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Add realistic CCTV noise and artifacts
        noise = np.random.normal(0, 20, security.shape).astype(np.int16)
        security = np.clip(security.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add horizontal scan lines (interlacing effect)
        for i in range(0, height, 2):
            security[i, :] = np.clip(security[i, :].astype(np.int16) - 15, 0, 255).astype(np.uint8)
        
        # Add motion blur lines randomly (typical of older tape cameras)
        for _ in range(2):
            blur_y = np.random.randint(0, height)
            blur_height = np.random.randint(5, 15)
            if blur_y + blur_height < height:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                security[blur_y:blur_y+blur_height, :] = cv2.filter2D(
                    security[blur_y:blur_y+blur_height, :], -1, kernel
                )
        
        # Add subtle vignette effect (darkened edges) without darkening too much
        rows, cols = security.shape
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols/2.5)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/2.5)
        
        # Generate vignette mask
        resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = resultant_kernel / resultant_kernel.max()
        mask = (mask * 200 + 55).astype(np.uint8)  # Scale to keep values between 55-255
        
        security = cv2.multiply(security, mask)
        
        # Convert back to BGR for final output
        security_bgr = cv2.cvtColor(security, cv2.COLOR_GRAY2BGR)
        
        # Add timestamp with realistic CCTV formatting
        from datetime import datetime
        current_time = datetime.now()
        timestamp = current_time.strftime("%H:%M:%S %m/%d/%Y")
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        # Add black background for timestamp
        (text_width, text_height), baseline = cv2.getTextSize(timestamp, font, font_scale, font_thickness)
        padding = 10
        cv2.rectangle(security_bgr, 
                     (5, height - text_height - padding - 5),
                     (text_width + padding + 5, height - 5),
                     (0, 0, 0), -1)
        
        # Add white timestamp text
        cv2.putText(security_bgr, timestamp, 
                   (10, height - 10), 
                   font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # Add camera info/ID
        camera_id = "CAM-01"
        cv2.putText(security_bgr, camera_id,
                   (10, 25),
                   font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add REC indicator (recording)
        rec_color = (0, 0, 255)  # Red
        cv2.circle(security_bgr, (width - 30, 25), 8, rec_color, -1)
        cv2.putText(security_bgr, "REC",
                   (width - 55, 30),
                   font, 0.5, rec_color, 1, cv2.LINE_AA)
        
        # Add resolution indicator
        res_text = f"{width}x{height}"
        cv2.putText(security_bgr, res_text,
                   (width - 150, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Add random tracking glitches
        if np.random.random() > 0.7:
            glitch_y = np.random.randint(0, height)
            glitch_height = np.random.randint(3, 8)
            glitch_x_shift = np.random.randint(-5, 5)
            if glitch_y + glitch_height < height:
                security_bgr[glitch_y:glitch_y+glitch_height, :] = np.roll(
                    security_bgr[glitch_y:glitch_y+glitch_height, :], glitch_x_shift, axis=1
                )
        
        return security_bgr
    
    def technique_film_burn(self, image):
        """Technique 19: Film Burn Transition"""
        height, width = image.shape[:2]
        film_burn = image.copy()
        light_leak = np.zeros_like(image)
        
        for i in range(height):
            intensity = int(255 * (1 - abs(i - height/2) / (height/2)))
            light_leak[i, :] = [0, intensity//2, intensity]
        
        for _ in range(5):
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            radius = np.random.randint(20, min(width, height)//4)
            
            for y in range(max(0, center_y - radius), min(height, center_y + radius)):
                for x in range(max(0, center_x - radius), min(width, center_x + radius)):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < radius:
                        darkness = int(255 * (1 - dist/radius) * 0.7)
                        film_burn[y, x] = np.clip(film_burn[y, x] - darkness, 0, 255)
        
        film_burn = cv2.addWeighted(film_burn, 0.8, light_leak, 0.3, 0)
        grain = np.random.normal(0, 10, film_burn.shape).astype(np.uint8)
        film_burn = cv2.add(film_burn, grain)
        return self.technique_vignette(film_burn)
    
    def technique_embroidery(self, image):
        """Technique 20: Embroidery Pattern"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((2, 2), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=1)
        embroidery = image.copy()
        colored_edges = np.zeros_like(image)
        colored_edges[thick_edges > 0] = [0, 0, 255]
        reduced_colors = cv2.convertScaleAbs(image, alpha=0.7, beta=30)
        embroidery = cv2.addWeighted(reduced_colors, 0.9, colored_edges, 0.4, 0)
        texture = np.random.normal(0, 5, embroidery.shape).astype(np.uint8)
        return cv2.add(embroidery, texture)
    
    # ===========================================
    # COMPUTER VISION TECHNIQUES (21-25)
    # ===========================================
    
    def technique_image_stitching(self, image):
        """Technique 21: Panorama Stitching (simulated)"""
        height, width = image.shape[:2]
        stitched = image.copy()
        
        # Create panorama-like effect by creating a tiled layout
        # Split image into sections
        sections = []
        num_sections = 3
        section_width = width // num_sections
        
        # Apply slight variations to each section to simulate different exposures
        for i in range(num_sections):
            x_start = i * section_width
            x_end = x_start + section_width if i < num_sections - 1 else width
            
            section = stitched[:, x_start:x_end].copy()
            
            # Apply slight perspective warp to each section
            if i == 0:
                # Left section - slight rotation
                angle = np.random.uniform(-5, 0)
            elif i == 1:
                # Middle section - straight
                angle = 0
            else:
                # Right section - slight rotation
                angle = np.random.uniform(0, 5)
            
            # Apply rotation
            center = (section.shape[1] // 2, section.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            section = cv2.warpAffine(section, M, (section.shape[1], section.shape[0]))
            
            # Apply different brightness to simulate exposure bracketing
            brightness_factor = 0.9 + (i * 0.1)
            section = cv2.convertScaleAbs(section, alpha=brightness_factor, beta=0)
            
            stitched[:, x_start:x_end] = section
        
        # Add blend lines where sections meet (seams)
        for i in range(1, num_sections):
            x_seam = i * section_width
            # Add a subtle gradient at seams
            blend_width = 15
            for dx in range(blend_width):
                alpha = (dx + 1) / (blend_width + 1)
                x_left = x_seam - blend_width + dx
                x_right = x_seam + dx
                if x_left >= 0 and x_right < width:
                    stitched[:, x_seam - blend_width + dx] = cv2.addWeighted(
                        stitched[:, x_left], 1 - alpha,
                        stitched[:, x_right], alpha, 0
                    )
        
        # Add panorama grid overlay
        grid_color = (100, 150, 100)
        for i in range(num_sections):
            x = i * section_width
            cv2.line(stitched, (x, 0), (x, height), grid_color, 2)
        
        # Add panorama label
        cv2.putText(stitched, "Panorama Stitching", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        
        # Show number of sections
        cv2.putText(stitched, f"Sections: {num_sections}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
        
        return stitched
    
    def technique_background_subtraction(self, image):
        """Technique 22: Background Subtraction"""
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations for better foreground detection
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Use adaptive thresholding for better foreground detection
        _, foreground_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Create background mask
        background_mask = cv2.bitwise_not(foreground_mask)
        
        # Create blurred background
        background = cv2.GaussianBlur(image, (25, 25), 0)
        
        # Extract foreground and background
        foreground = cv2.bitwise_and(image, image, mask=foreground_mask)
        background_result = cv2.bitwise_and(background, background, mask=background_mask)
        
        # Combine foreground and blurred background
        result = cv2.add(foreground, background_result)
        
        # Find and draw contours of detected objects
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort contours by area and keep the largest ones
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            
            # Draw all major contours
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 100:  # Only draw significant contours
                    # Draw filled contour with semi-transparency effect
                    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                    
                    # Get bounding box and draw it
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Add area label
                    if i == 0:  # Label the largest object
                        cv2.putText(result, f"Area: {int(area)}", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add statistics panel
        num_objects = len([c for c in contours if cv2.contourArea(c) > 100]) if contours else 0
        cv2.putText(result, f"Objects Detected: {num_objects}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add legend
        cv2.putText(result, "Green: Foreground Edge | Blue: Bounding Box",
                   (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return result
    
    def technique_image_compression(self, image, quality=50):
        """Technique 23: Image Compression"""
        compressed = image.copy()
        
        # Apply multiple rounds of compression/decompression to show artifacts
        for compression_round in range(3):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality - (compression_round * 10)]
            encode_param[1] = max(encode_param[1], 10)  # Minimum quality 10
            result, encimg = cv2.imencode('.jpg', compressed, encode_param)
            
            if result:
                compressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        
        # Add blockiness effect (8x8 blocks typical of JPEG)
        block_size = 8
        h, w = compressed.shape[:2]
        
        # Create visible block artifacts
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                # Get the block
                block = compressed[y:y+block_size, x:x+block_size]
                # Add subtle edge darkening to show block boundaries
                if block.shape[0] > 0 and block.shape[1] > 0:
                    # Darken edges
                    block[0, :] = (block[0, :] * 0.85).astype(np.uint8)
                    block[-1, :] = (block[-1, :] * 0.85).astype(np.uint8)
                    block[:, 0] = (block[:, 0] * 0.85).astype(np.uint8)
                    block[:, -1] = (block[:, -1] * 0.85).astype(np.uint8)
        
        # Add compression info overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = f"JPEG Quality: {quality}%"
        cv2.putText(compressed, info_text, (10, 30), font, 0.7, (0, 0, 255), 2)
        
        # Calculate compression ratio
        original_size = image.nbytes
        encode_param_final = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg_final = cv2.imencode('.jpg', image, encode_param_final)
        compressed_size = len(encimg_final) if hasattr(encimg_final, '__len__') else 1000
        
        ratio = (compressed_size / original_size) * 100
        ratio_text = f"Size: {ratio:.1f}%"
        cv2.putText(compressed, ratio_text, (10, 60), font, 0.6, (0, 100, 255), 1)
        
        # Show file size metrics
        size_text = f"Est. {compressed_size // 1024}KB"
        cv2.putText(compressed, size_text, (10, 85), font, 0.5, (100, 100, 255), 1)
        
        return compressed
    
    def technique_style_transfer(self, image):
        """Technique 24: Style Transfer"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        hsv[:,:,1] = cv2.multiply(hsv[:,:,1], 1.5)
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = cv2.equalizeHist(lab_planes[0])
        lab = cv2.merge(lab_planes)
        
        saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        contrasted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        styled = cv2.addWeighted(saturated, 0.6, contrasted, 0.4, 0)
        
        kernel_size = random.choice([3, 5, 7])
        styled = cv2.bilateralFilter(styled, kernel_size, 75, 75)
        
        texture = np.random.normal(128, 10, styled.shape).astype(np.uint8)
        styled = cv2.addWeighted(styled, 0.9, texture, 0.1, 0)
        
        cv2.putText(styled, "Style Transfer", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return styled
    
    def technique_optical_flow(self, image):
        """Technique 25: Optical Flow (simulated)"""
        height, width = image.shape[:2]
        flow_display = image.copy()
        
        # Create a more realistic optical flow using edge detection
        gray = cv2.cvtColor(flow_display, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Create denser motion grid
        grid_size = 15
        points = []
        
        for y in range(grid_size, height - grid_size, grid_size):
            for x in range(grid_size, width - grid_size, grid_size):
                points.append((x, y))
        
        # Create flow field based on image gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate motion vectors for each point
        for (x, y) in points:
            if 0 <= y < height and 0 <= x < width:
                # Get gradient at this point
                gx = sobel_x[y, x]
                gy = sobel_y[y, x]
                
                # Calculate angle and magnitude from gradients
                magnitude = np.sqrt(gx**2 + gy**2)
                
                if magnitude > 10:
                    # Normalize and scale
                    angle = np.arctan2(gy, gx)
                    motion_magnitude = min(magnitude / 10, 25)
                else:
                    # Random motion if no strong gradient
                    angle = np.random.uniform(0, 2 * np.pi)
                    motion_magnitude = np.random.uniform(5, 15)
                
                # Calculate end point
                x2 = int(x + motion_magnitude * np.cos(angle))
                y2 = int(y + motion_magnitude * np.sin(angle))
                
                # Clamp to image bounds
                x2 = np.clip(x2, 0, width - 1)
                y2 = np.clip(y2, 0, height - 1)
                
                # Draw arrow with color based on magnitude
                color_intensity = int(np.clip(motion_magnitude * 10, 0, 255))
                color = (color_intensity, 255 - color_intensity // 2, 100)
                
                cv2.arrowedLine(flow_display, (x, y), (x2, y2), 
                               color, 2, tipLength=0.4)
                cv2.circle(flow_display, (x, y), 3, (255, 100, 0), -1)
        
        # Create detailed heatmap from gradient magnitude
        magnitude_full = np.sqrt(sobel_x**2 + sobel_y**2)
        heatmap = cv2.normalize(magnitude_full, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply Gaussian blur for smoothness
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Apply color map
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend with original
        flow_display = cv2.addWeighted(flow_display, 0.65, heatmap_color, 0.35, 0)
        
        # Add information panel
        cv2.putText(flow_display, "Optical Flow Analysis", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(flow_display, "Motion: Green=Arrows, Red-Blue=Intensity", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add legend
        cv2.line(flow_display, (width - 200, height - 50), (width - 100, height - 50), 
                (0, 255, 0), 3)
        cv2.putText(flow_display, "Low Motion", (width - 95, height - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.line(flow_display, (width - 200, height - 25), (width - 100, height - 25),
                (255, 100, 0), 3)
        cv2.putText(flow_display, "High Motion", (width - 95, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
        
        return flow_display
    
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
            'canny_edge': self.technique_canny_edge,
            'anime_style': self.technique_anime_style,
            'sepia_tone': self.technique_sepia_tone,
            'pencil_sketch': self.technique_pencil_sketch,
            'sharpen': self.technique_sharpen,
            'edge_detection': self.technique_edge_detection,
            'binary_threshold': self.technique_binary_threshold,
            'emboss': self.technique_emboss,
            'oil_painting': self.technique_oil_painting,
            'cartoon': self.technique_cartoon,
            'hdr': self.technique_hdr,
            'watercolor': self.technique_watercolor,
            'vignette': self.technique_vignette,
            'movie_poster': self.technique_movie_poster,
            'album_cover': self.technique_album_cover,
            'vhs_effect': self.technique_vhs_effect,
            'pointillism': self.technique_pointillism,
            'security_camera': self.technique_security_camera,
            'film_burn': self.technique_film_burn,
            'embroidery': self.technique_embroidery,
            'image_stitching': self.technique_image_stitching,
            'background_subtraction': self.technique_background_subtraction,
            'image_compression': self.technique_image_compression,
            'style_transfer': self.technique_style_transfer,
            'optical_flow': self.technique_optical_flow
        }
        
        if technique_name in technique_methods:
            # Special handling for techniques with parameters
            if technique_name == 'image_compression':
                return technique_methods[technique_name](image, quality=50)
            elif technique_name == 'vignette':
                return technique_methods[technique_name](image, vignette_strength=0.7)
            else:
                return technique_methods[technique_name](image)
        else:
            print(f"  Warning: Unknown technique '{technique_name}'")
            return image
    
    def process_single_image(self, image_path):
        """
        Apply all techniques to a single image.
        
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