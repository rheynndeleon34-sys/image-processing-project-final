#!/usr/bin/env python3

import sys
from pathlib import Path
from src.image_processor import ImageProcessor

def display_banner():
    """Display enhanced program banner"""
    banner = """
    ===================================================
            ADVANCED IMAGE PROCESSING PIPELINE
    ===================================================
    """
    print(banner)

def display_all_techniques():
    """Display all 25 implemented techniques"""
    categories = {
        "Basic Techniques": [
            ("1", "Canny Edge Detection", "Detect edges with Canny algorithm"),
            ("2", "Anime Animation Style", "Transform into anime/cartoon characters"),
            ("3", "Sepia Tone", "Vintage brown filter"),
            ("4", "Pencil Sketch", "Convert to pencil drawing"),
            ("5", "Image Sharpening", "Enhance details"),
            ("6", "Enhanced Edge Detection", "Multi-method edge detection"),
            ("7", "Binary Threshold", "Pure black and white"),
            ("8", "Emboss Effect", "3D relief effect")
        ],
        "Advanced Techniques": [
            ("9", "Oil Painting", "Oil painting artistic effect"),
            ("10", "Cartoon Effect", "Cartoon style"),
            ("11", "HDR Effect", "High dynamic range enhancement"),
            ("12", "Watercolor", "Watercolor painting effect"),
            ("13", "Vignette", "Darkened corners effect")
        ],
        "Artistic Effects": [
            ("14", "Movie Poster", "Cinematic poster with title and credits"),
            ("15", "Album Cover", "Music album cover design"),
            ("16", "VHS Effect", "Analog tape degradation"),
            ("17", "Pointillism", "Classic painting technique"),
            ("18", "Security Camera", "Surveillance camera effect"),
            ("19", "Film Burn", "Cinematic light leak effect"),
            ("20", "Embroidery", "Stitching pattern effect")
        ],
        "Computer Vision Techniques": [
            ("21", "Panorama Stitching", "Image stitching for panoramas"),
            ("22", "Background Subtraction", "Remove/change background"),
            ("23", "Image Compression", "JPEG compression simulation"),
            ("24", "Style Transfer", "Apply artistic styles"),
            ("25", "Optical Flow", "Motion vector visualization")
        ]
    }
    
    print("IMPLEMENTED TECHNIQUES (25 Total):")
    print("=" * 70)
    
    for category, techniques in categories.items():
        print(f"\n{category}:")
        print("-" * 70)
        for num, name, desc in techniques:
            print(f"{num:>3}. {name:<25} - {desc}")
    
    print("=" * 70)

def display_usage():
    """Display usage instructions"""
    print("\nUSAGE:")
    print("  python main.py [input_dir] [output_dir] [--techniques TECH1,TECH2,...]")
    print("\nEXAMPLES:")
    print("  python main.py                        # Use default directories")
    print("  python main.py input output           # Specify directories")
    print("  python main.py --techniques canny_edge,sepia,movie_poster  # Specific techniques")
    print("\nOPTIONS:")
    print("  --techniques TECH1,TECH2,...  Apply only specified techniques")
    print("  --all                         Apply all 25 techniques (default)")
    print("  --basic                       Apply only basic techniques (1-8)")
    print("  --advanced                    Apply only advanced techniques (9-13)")
    print("  --artistic                    Apply only artistic effects (14-20)")
    print("  --cv                          Apply only computer vision techniques (21-25)")

def parse_arguments():
    """Parse command line arguments"""
    args = {
        'input_dir': 'input',
        'output_dir': 'output',
        'techniques': None  # None means all techniques
    }
    
    # Simple argument parsing
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg in ['--techniques', '-t']:
            if i + 1 < len(sys.argv):
                args['techniques'] = sys.argv[i + 1].split(',')
                i += 2
            else:
                print(f"Error: --techniques requires a comma-separated list")
                sys.exit(1)
        
        elif arg == '--basic':
            args['techniques'] = [
                'canny_edge', 'anime_style', 'sepia_tone', 'pencil_sketch',
                'sharpen', 'edge_detection', 'binary_threshold', 'emboss'
            ]
            i += 1
        
        elif arg == '--advanced':
            args['techniques'] = [
                'oil_painting', 'cartoon', 'hdr', 'watercolor', 'vignette'
            ]
            i += 1
        
        elif arg == '--artistic':
            args['techniques'] = [
                'movie_poster', 'album_cover', 'vhs_effect', 'pointillism',
                'security_camera', 'film_burn', 'embroidery'
            ]
            i += 1
        
        elif arg == '--cv':
            args['techniques'] = [
                'image_stitching', 'background_subtraction', 'image_compression',
                'style_transfer', 'optical_flow'
            ]
            i += 1
        
        elif arg == '--all':
            args['techniques'] = None  # All techniques
            i += 1
        
        elif arg == '--help' or arg == '-h':
            display_usage()
            sys.exit(0)
        
        elif arg.startswith('-'):
            print(f"Warning: Unknown option {arg}")
            i += 1
        
        else:
            # Positional arguments
            if 'input_dir' in args and args['input_dir'] == 'input':
                args['input_dir'] = arg
            elif 'output_dir' in args and args['output_dir'] == 'output':
                args['output_dir'] = arg
            i += 1
    
    return args

def main():
    """Main function to run the enhanced image processor"""
    # Display banner
    display_banner()
    
    # Display all techniques
    display_all_techniques()
    
    # Parse arguments
    args = parse_arguments()
    
    print(f"\nCONFIGURATION:")
    print(f"  Input Directory:  {args['input_dir']}")
    print(f"  Output Directory: {args['output_dir']}")
    
    if args['techniques']:
        print(f"  Techniques:       {len(args['techniques'])} specified")
        if len(args['techniques']) <= 10:
            print(f"                    {', '.join(args['techniques'])}")
    else:
        print(f"  Techniques:       All 25 techniques")
    
    print()
    
    # Initialize the image processor
    try:
        processor = ImageProcessor(args['input_dir'], args['output_dir'])
    except Exception as e:
        print(f"Error initializing processor: {e}")
        return
    
    # Process all images
    print("STARTING PROCESSING...")
    print("=" * 70)
    
    success_count, total_count, stats = processor.process_all_images(args['techniques'])
    
    # Display results
    print("=" * 70)
    print("\nPROCESSING COMPLETE!")
    print("=" * 70)
    
    if success_count > 0:
        print(f"✅ SUCCESSFULLY PROCESSED: {success_count}/{total_count} images")
        print(f"📁 OUTPUT LOCATION: {args['output_dir']}/")
        print(f"📊 FILES CREATED: {stats['files_created']} total files")
        
        # Calculate versions per image
        versions_per_image = stats['files_created'] // success_count if success_count > 0 else 0
        print(f"   ({versions_per_image} versions per image: original + {versions_per_image - 1} techniques)")
        
        # Show sample output
        output_path = Path(args['output_dir'])
        if output_path.exists():
            png_files = list(output_path.glob("*.png"))
            if png_files:
                print(f"\n SAMPLE OUTPUT FILES:")
                # Group by image name
                from collections import defaultdict
                grouped = defaultdict(list)
                for file in png_files:
                    base = '_'.join(file.stem.split('_')[:-1]) if '_' in file.stem else file.stem
                    grouped[base].append(file.name)
                
                # Display first 3 images with their techniques
                for i, (base_name, files) in enumerate(list(grouped.items())[:3]):
                    print(f"\n  Image: {base_name}")
                    tech_names = [f.split('_')[-1].replace('.png', '') for f in files[:5]]
                    print(f"    Techniques: {', '.join(tech_names[:5])}")
                    if len(tech_names) > 5:
                        print(f"                ... and {len(tech_names) - 5} more")
        
        print(f"\n🎨 TECHNIQUES APPLIED: {stats['total_techniques']}")
        
        # Suggest next steps
        print("\n NEXT STEPS:")
        print("  1. View results in the output folder")
        print("  2. Try different technique groups: --basic, --advanced, --artistic, --cv")
        
    else:
        print("No images were processed.")
        print("\n TROUBLESHOOTING:")
        print("  1. Create an 'input' folder")
        print("  2. Add image files (jpg, png, etc.)")
        print("  3. Or run: python demo/create_demo_images.py")
        print("  4. Then run: python main.py")
    
    print("\n" + "=" * 70)
    print("Thank you for using our Image Processing Pipeline!")
    print("=" * 70)

if __name__ == "__main__":
    main()