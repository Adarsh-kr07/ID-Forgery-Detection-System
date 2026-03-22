import exifread

def get_metadata(file):
    tags = exifread.process_file(file)
    return {str(k): str(v) for k, v in tags.items()}