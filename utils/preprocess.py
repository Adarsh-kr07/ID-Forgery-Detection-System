# utils/preprocess.py
def validate_image(image):
    return image.format in ["JPEG", "PNG", "JPG"]