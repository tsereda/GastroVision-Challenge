from PIL import Image
import os

def find_smallest_resolution(start_path):
    min_area = float('inf')
    smallest_file = None
    smallest_dims = (0, 0)
    
    # Walk through the directory
    for root, dirs, files in os.walk(start_path):
        for name in files:
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, name)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        area = width * height
                        
                        # Check if this is the new smallest area
                        if area < min_area:
                            min_area = area
                            smallest_file = file_path
                            smallest_dims = (width, height)
                            
                except Exception as e:
                    print(f"Error reading {name}: {e}")

    if smallest_file:
        print(f"Smallest resolution image: {smallest_file}")
        print(f"Dimensions: {smallest_dims[0]}x{smallest_dims[1]}")
        print(f"Total Pixels: {min_area}")
    else:
        print("No images found.")

# Update the path to match your folder
dir_path = "Gastrovision Challenge dataset/Training data/Erythema"
find_smallest_resolution(dir_path)