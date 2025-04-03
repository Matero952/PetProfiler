from PIL import Image
import os

# Paths
DOGS_FOLDER = "/home/mateo/Github/PetProfiler/updated_dataset/dog/overlays"  # Folder containing cropped dog images
BACKGROUNDS_FOLDER = "/home/mateo/Github/PetProfiler/updated_dataset/dog/bgs"  # Folder with background images
OUTPUT_FOLDER = "/home/mateo/Github/PetProfiler/updated_dataset/dog/done"  # Folder for generated images

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Get list of images
dog_images = [f for f in os.listdir(DOGS_FOLDER) if f.endswith(".png")]
background_images = [f for f in os.listdir(BACKGROUNDS_FOLDER) if f.endswith((".png", ".jpg", ".jpeg"))]


def remove_white_background(image_path):
    # Open the image and convert to RGBA (if not already)
    image = Image.open(image_path).convert("RGBA")
    data = image.getdata()

    # Create a new list for the modified pixels
    new_data = []
    for item in data:
        # If the pixel is exactly white, make it transparent
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append((255, 255, 255, 0))  # transparent pixel
        else:
            new_data.append(item)

    # Update the image with the modified data
    image.putdata(new_data)
    return image


def overlay_image(background, overlay, pos_x, pos_y):
    background.paste(overlay, (pos_x, pos_y), overlay)  # Uses transparency
    return background

# Define the positions for overlaying the dog images
positions = [(10, 300)]

# Process each dog image with each background
counter = 0
for dog_image in dog_images:
    dog_path = os.path.join(DOGS_FOLDER, dog_image)
    dog_image = remove_white_background(dog_path)  # Remove white background from dog image

    for bg_img in background_images:
        bg_path = os.path.join(BACKGROUNDS_FOLDER, bg_img)
        bg_image = Image.open(bg_path).convert("RGBA")
        background_resized = bg_image.resize((640, 640))  # Resize background to 640x640
        dog_resized = dog_image.resize((640, 640))  # Resize dog image to 640x640

        # Overlay dog images on each background at the specified positions
        for pos in positions:
            pos_x, pos_y = pos
            overlayed = overlay_image(background_resized.copy(), dog_resized.copy(), pos_x, pos_y)
            output_path = os.path.join(OUTPUT_FOLDER, f"dog_overlay_{counter}_sec2.png")
            overlayed.save(output_path)
            print(f"{dog_image} overlayed on {bg_img} at position ({pos_x}, {pos_y})")
            counter += 1