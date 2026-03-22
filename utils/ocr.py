import numpy as np
import cv2
import tempfile
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Load model once when file is imported
model = ocr_predictor(pretrained=True)

def extract_text(image):
    # Save numpy array as temporary image file
    if isinstance(image, np.ndarray):
        # Save to a temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        tmp.close()
        
        cv2.imwrite(tmp_path, image)
        
        # Pass file path to DocTR
        doc = DocumentFile.from_images([tmp_path])
        
        # Clean up temp file
        os.remove(tmp_path)
    else:
        doc = DocumentFile.from_images([image])

    # Run OCR
    result = model(doc)

    # Extract all text
    extracted_text = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])
                extracted_text.append(line_text)

    return "\n".join(extracted_text)