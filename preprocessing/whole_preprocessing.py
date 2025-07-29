from noise_filter import apply_correction

print("Starting preprocessing...")

image_path = "ts.png"
for i in range(1, 2):
    enhanced_image = apply_correction(image_path)
    image_path ="test_output.png"

#      \/ add land segmentation here \/

