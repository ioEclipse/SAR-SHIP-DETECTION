from noise_filter import apply_correction

print("Starting preprocessing...")
#      \/ add land segmentation here \/



#    this is the noise filter
image_path = "ts.png"
for i in range(1, 2):
    enhanced_image = apply_correction(image_path)
    image_path ="test_output.png"



