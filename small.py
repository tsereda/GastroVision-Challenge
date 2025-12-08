import os

def find_smallest_file(start_path):
    smallest_size = float('inf')
    smallest_file = None
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(start_path):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                # Get file size in bytes
                size = os.path.getsize(file_path)
                
                if size < smallest_size:
                    smallest_size = size
                    smallest_file = file_path
            except OSError as e:
                print(f"Could not access {file_path}: {e}")

    if smallest_file:
        print(f"Smallest file: {smallest_file}")
        print(f"Size: {smallest_size} bytes ({smallest_size/1024:.2f} KB)")
    else:
        print("No files found.")

# The path from your screenshot
dir_path = "GastroVision-Challenge/Gastrovision Challenge dataset/Training data/Erythema"
find_smallest_file(dir_path)