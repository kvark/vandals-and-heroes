from PIL import Image
import numpy as np

world = 'boozina'
# Source data was produced by vange-rs converter as follows
# cargo run --bin convert --release c:\GOG\Games\Vangers\data\thechain\{world}\world.ini output\boozina.ron

# Load the first and second images
image1 = Image.open('C:\\Code\\Vangers\\vange-rs\\output\\height.png')  # Image from which we'll take the red channel
image2 = Image.open('C:\\Code\\Vangers\\vange-rs\\output\\material_lo.png')  # Image to which we'll add the red channel as alpha

# Ensure both images are in RGBA mode (if they are not)
image1 = image1.convert("RGBA")
image2 = image2.convert("RGBA")

# Extract the red channel from the first image
red_channel = np.array(image1)[:, :, 0]  # The first channel is the red channel

# Extract the other channels of the second image (keeping RGB)
image2_data = np.array(image2)
image2_data[:, :, 3] = red_channel  # Replace the alpha channel with the red channel from image1

# Convert the modified numpy array back to an image
output_image = Image.fromarray(image2_data)

# Save or show the final image
output_image.save(f'data/maps/{world}/map.png')
output_image.show()
