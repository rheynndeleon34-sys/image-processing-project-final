import cv2
import numpy as np
from pathlib import Path

def create_color_gradient():
    """Create a colorful gradient image"""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Create RGB gradient
    for y in range(400):
        # Blue increases top to bottom
        img[y, :, 0] = y * 255 // 400
        
        # Green is constant
        img[y, :, 1] = 128
        
        # Red decreases top to bottom
        img[y, :, 2] = 255 - (y * 255 // 400)
    
    # Add text
    cv2.putText(img, "Gradient", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    return img

def create_shapes_image():
    """Create image with geometric shapes"""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Draw shapes
    cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), -1)  # Red rectangle
    cv2.circle(img, (450, 150), 80, (0, 255, 0), -1)  # Green circle
    cv2.line(img, (300, 300), (500, 350), (255, 0, 0), 5)  # Blue line
    cv2.ellipse(img, (150, 300), (100, 50), 45, 0, 360, (255, 255, 0), -1)  # Yellow ellipse
    
    # Add text
    cv2.putText(img, "Shapes", (250, 380), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    return img

def create_portrait_like():
    """Create a simulated portrait"""
    img = np.random.randint(150, 200, (500, 400, 3), dtype=np.uint8)
    
    # Add face oval
    cv2.ellipse(img, (200, 200), (120, 150), 0, 0, 360, (180, 150, 130), -1)
    
    # Add eyes
    cv2.circle(img, (160, 180), 20, (255, 255, 255), -1)
    cv2.circle(img, (240, 180), 20, (255, 255, 255), -1)
    cv2.circle(img, (160, 180), 8, (0, 0, 0), -1)
    cv2.circle(img, (240, 180), 8, (0, 0, 0), -1)
    
    # Add mouth
    cv2.ellipse(img, (200, 260), (60, 30), 0, 0, 180, (200, 100, 100), -1)
    
    # Add text
    cv2.putText(img, "Portrait", (150, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    return img

def create_nature_scene():
    """Create a simple nature scene"""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Sky (blue gradient)
    for y in range(200):
        img[y, :, 0] = 150 + y // 2  # Blue
        img[y, :, 1] = 200 + y // 3  # Green
        img[y, :, 2] = 255  # Red
    
    # Ground
    img[200:, :, 1] = 100  # Green ground
    img[200:, :, 0] = 50   # Some blue in ground
    img[200:, :, 2] = 50   # Some red in ground
    
    # Sun
    cv2.circle(img, (500, 80), 40, (0, 255, 255), -1)
    
    # Tree
    cv2.rectangle(img, (150, 200), (170, 350), (100, 50, 0), -1)  # Trunk
    cv2.circle(img, (160, 150), 60, (0, 100, 0), -1)  # Leaves
    
    # Add text
    cv2.putText(img, "Nature", (250, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    return img

def main():
    """Create demo images in the input folder"""
    # Create input directory if it doesn't exist
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    
    print("Creating demo images...")
    
    # Create and save images
    images = [
        ("gradient.jpg", create_color_gradient()),
        ("shapes.png", create_shapes_image()),
        ("portrait.jpg", create_portrait_like()),
        ("nature_scene.png", create_nature_scene())
    ]
    
    for filename, image in images:
        filepath = input_dir / filename
        cv2.imwrite(str(filepath), image)
        print(f"  Created: {filename} ({image.shape[1]}x{image.shape[0]})")
    
    print(f"\n✅ Demo images created in '{input_dir}/' folder")
    print("\nYou can now run:")
    print("  python main.py")

if __name__ == "__main__":
    main()