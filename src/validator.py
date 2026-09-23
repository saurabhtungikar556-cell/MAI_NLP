import pytesseract
import cv2
import numpy as np
from PIL import Image
import io

class CircuitValidator:
    def __init__(self):
        # Vocabulary that strongly suggests "Not a Circuit"
        self.negative_vocab = ["plot", "graph", "performance", "error", "accuracy", "loss", "setup", "apparatus", "schema"]
        # Vocabulary that strongly suggests "Circuit"
        self.positive_vocab = ["circuit", "gate", "qubit", "algorithm", "ansatz", "|0>", "H", "CNOT"]
        
    def validate(self, pil_image, caption_text):
        """
        Returns (is_valid: bool, reason: str)
        """
        # 1. Caption Filter (Fastest)
        if not caption_text:
            return False, "No caption found"
            
        caption_lower = caption_text.lower()
        # If caption explicitly says "Plot of...", reject.
        if any(w in caption_lower for w in ["plot of", "dependence of", "function of"]):
            return False, "Caption implies plot"

        # 2. OCR Content Filter (Slower but accurate)
        # We look for text INSIDE the image.
        try:
            ocr_text = pytesseract.image_to_string(pil_image)
        except:
            ocr_text = ""
            
        # Check for axis labels (typical of plots)
        if any(x in ocr_text for x in ["Hz", "nm", "Error (%)", "Time (s)", "Probability"]):
            return False, "OCR found plot axis labels"

        # Check for Circuit Symbols
        # Circuit diagrams are usually sparse text. 
        has_circuit_terms = any(x in ocr_text for x in ["|0", "|1", "H", "q_", "measure"])
        
        # 3. Visual Density Check
        # Convert to CV2 format for analysis
        open_cv_image = np.array(pil_image) 
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        
        # Edge density: Plots usually have high edge density (grids, noise).
        # Circuits are mostly empty space (white) with straight lines.
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        if edge_density > 0.15: # Threshold to be tuned
            return False, "Image too complex/dense (likely a plot or photo)"
            
        # If we passed negative filters, check for positive affirmation
        if has_circuit_terms or any(w in caption_lower for w in self.positive_vocab):
            return True, "Passed filters"
            
        return False, "Ambiguous content"