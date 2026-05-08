from PIL import Image, ImageDraw, ImageFont
import os

# Create labels directory
output_dir = "labels"
os.makedirs(output_dir, exist_ok=True)

# Define buildings and their labels
buildings = {
    "pharmacy": "Pharmacy",
    "firefighting_center": "Fire Station",
    "supermarket": "Supermarket",
    "restaurant": "Restaurant",
    "house_1": "House 1",
    "house_2": "House 2",
    "house_3": "House 3",
    "house_4": "House 4",
    "house_5": "House 5",
    "docking_station": "Docking",
}

# Image settings
width, height = 512, 256
background_color = (0, 0, 0)  # Black background
text_color = (255, 255, 255)  # White text
font_size = 80

# Try to use a system font, fallback to default
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
except:
    font = ImageFont.load_default()

for building_key, building_name in buildings.items():
    # Create image
    img = Image.new('RGBA', (width, height), background_color + (255,))
    draw = ImageDraw.Draw(img)
    
    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), building_name, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text
    draw.text((x, y), building_name, font=font, fill=text_color)
    
    # Save image
    filename = os.path.join(output_dir, f"{building_key}.png")
    img.save(filename)
    print(f"Created {filename}")

print("All label images created successfully!")
