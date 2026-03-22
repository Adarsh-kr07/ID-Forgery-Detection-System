import numpy as np
import cv2
import tempfile
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


model = ocr_predictor(pretrained=True)

def extract_text(image):
    
    if isinstance(image, np.ndarray):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        tmp.close()
        
        cv2.imwrite(tmp_path, image)
        
        doc = DocumentFile.from_images([tmp_path])
        
        os.remove(tmp_path)
    else:
        doc = DocumentFile.from_images([image])

    # Running OCR
    result = model(doc)

    # Extract all text from uploaded image
    extracted_text = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])
                extracted_text.append(line_text)

    return "\n".join(extracted_text)