import cv2, math, re, pytesseract
import numpy as np
from config import CONFIG

class CircuitValidator: 
    def __init__(self):
        self.mandatory_keywords = ["circuit", "quantum", "schematic"] 
        self.hard_negative_keywords = [ 
            "plot", "graph", "dependence", "vs.", "error rate", "comparison", "fidelity vs", "population", "curve", "setup", "apparatus", "schema", "laser", "beam splitter", 
            "interferometer", "mirror", "lens", "detection", "mach-zehnder", "mzi", "spcm", "histogram", "axis", "axes", "pattern"
        ]
        self.soft_negative_keywords = [ 
            "simulation", "result", "performance", "time", "evolution", 
            "code", "pseudo-code", "python", "algorithm", "accuracy", "loss"
        ]
        self.ocr_circuit_tokens = [ 
            "|0>", "|1>", "q_", "q0", "CNOT", "H", "measure", "R_x", "P", "X", "Y", "Z", "SWAP", "ctrl",
            "|0⟩", "|1⟩", "|psi>", "|ψ>", "q[0]", "q[1]", "U3", "U2", "barrier", "measure_all", "R_y", "R_z"
        ]
        self.algo_keywords = [
            "shor", "grover", "deutsch", "jozsa", "bernstein", "vazirani", "simon", "hhl", "qaoa", "vqe", "uccsd", "qpe",  
            "teleportation", "superdense coding", "bb84", "e91", "qkd", "entanglement swapping", "magic state",
            "surface code", "repetition code", "steane", "toric code", "color code", "stabilizer code",
            "qft", "quantum fourier transform", "phase estimation", "amplitude amplification", "randomized benchmarking", "quantum walk", "swap test", "hadamard test"
        ]
        
    def _check_golden_phrase(self, caption_text): 
        if not caption_text: return False
        clean = caption_text.lower().replace('\n', ' ')
        if re.search(r'(?:fig\.?|figure).*?quantum\s+circuit', clean[:150]): return True
        return False
        
    def _check_proximity_keywords(self, caption_text): 
        if not caption_text: return False
        clean_text = caption_text.lower().replace('\n', ' ')
        words = clean_text.split()
        anchor_indices = [i for i, w in enumerate(words) if w.startswith("fig")]
        if not anchor_indices: return False 
        for idx in anchor_indices:
            start = max(0, idx - 10) 
            end = min(len(words), idx + 11) 
            window = words[start:end]
            if any(k in window for k in self.mandatory_keywords): return True
        return False
        
    def has_color_blocks(self, pil_image): 
        img = np.array(pil_image)[:, :, ::-1].copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 0]), np.array([179, 255, 255])) 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > CONFIG["COLOR_BLOCK_THRESHOLD"]:
                x, y, w, h = cv2.boundingRect(cnt)
                if min(w, h) > 15: return True
        return False
        
    def has_colored_text(self, pil_image): 
        img = np.array(pil_image)[:, :, ::-1].copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 30, 0]), np.array([179, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_color_blob_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if 5 < area < 500 and w < 100 and h < 50: 
                small_color_blob_count += 1
        if small_color_blob_count > 50: return True
        return False
        
    def has_gray_blocks(self, pil_image): 
        img = np.array(pil_image)
        img_bgr = img[:, :, ::-1].copy()
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([179, 40, 200])) 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bad_blob_area = 0
        image_area = img.shape[0] * img.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300: 
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                solidity = float(area) / rect_area 
                if solidity < 0.90: 
                    bad_blob_area += area
                if rect_area > 3000 and min(w, h) > 40: 
                    density = area / float(rect_area)
                    if density > 0.5: return True
        if bad_blob_area > (image_area * 0.05): 
            return True
        return False
        
    def has_curves(self, pil_image): 
        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150) 
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10) 
        line_mask = np.zeros_like(edges)
        if lines is not None:
            for line in lines: cv2.line(line_mask, (line[0][0], line[0][1]), (line[0][2], line[0][3]), 255, 3) 
        curves_only = cv2.bitwise_and(edges, cv2.bitwise_not(line_mask)) 
        cleaned_curves = cv2.morphologyEx(curves_only, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mass = np.count_nonzero(cleaned_curves)
        if mass > CONFIG["CURVE_THRESHOLD"]: return True
        return False
        
    def has_dense_grid(self, pil_image): 
        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        square_count = 0
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True) 
            if len(approx) == 4: 
                x, y, w, h = cv2.boundingRect(approx)
                ar = w / float(h)
                if 0.8 < ar < 1.2 and w > 20 and h > 20: square_count += 1
        if square_count > 6: return True
        return False
        
    def has_horizontal_wires(self, pil_image): 
        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=5) 
        has_long_wire = False
        img_width = img.shape[1]
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if abs(angle) < 5 or abs(angle) > 175: 
                    length = abs(x2 - x1)
                    if length > (img_width * 0.20):
                        has_long_wire = True
                        break 
        return has_long_wire
        
    def detect_plot_axes(self, pil_image): 
        img = np.array(pil_image)
        if len(img.shape) == 3: gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else: gray = img
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=pil_image.width*0.2, maxLineGap=10)
        has_b, has_l = False, False
        if lines is not None:
            for line in lines:
                if abs(line[0][1]-line[0][3])<5 and line[0][1] > pil_image.height*0.8: has_b = True 
                if abs(line[0][0]-line[0][2])<5 and line[0][0] < pil_image.width*0.2: has_l = True 
        return (has_b and has_l)
        
    def validate(self, pil_image, caption_text): 
        meta = {"gates": [], "algorithm": "unknown"}
        if pil_image is None: return False, "Empty", meta
        if pil_image.width == 0 or pil_image.height == 0: return False, "Empty Image", meta
        if any(w in caption_text.lower() for w in self.hard_negative_keywords):
             return False, "Rejected (Hard Keyword in Caption)", meta
        try: 
           ocr_text = pytesseract.image_to_string(pil_image, config='--psm 11')
        except: 
            ocr_text = ""
        optical_tokens = ["BS", "PBS", "HWP", "LP", "Detector", "mirror", "lens"]
        plot_tokens = ["MSE", "Iteration", "Counts", "Probability", "Fidelity", "Error", "vs.", "Axis"]
        if any(token in ocr_text for token in plot_tokens): return False, "Rejected (Plot terms in OCR)", meta
        if any(token in ocr_text for token in optical_tokens): return False, "Rejected (Optical Setup in OCR)", meta
        if self.detect_plot_axes(pil_image): return False, "Rejected (Axes Detected)", meta
        if self._check_golden_phrase(caption_text):
            found_tokens = [t for t in self.ocr_circuit_tokens if t in ocr_text]
            meta["gates"] = list(set(found_tokens))
            return True, "Valid (Golden Phrase)", meta
        if any(w in caption_text.lower() for w in self.soft_negative_keywords):
             return False, "Rejected (Soft Keyword in Caption)", meta
        if not self._check_proximity_keywords(caption_text):
            return False, "Rejected (Keywords missing)", meta
        if not self.has_horizontal_wires(pil_image): 
            return False, "Rejected (No horizontal wires)", meta
        if self.has_color_blocks(pil_image): return False, "Rejected (Color Block)", meta
        if self.has_colored_text(pil_image): return False, "Rejected (Syntax Highlight)", meta
        if self.has_gray_blocks(pil_image): return False, "Rejected (Gray Block)", meta
        if self.has_curves(pil_image): return False, "Rejected (Curves)", meta
        if self.has_dense_grid(pil_image): return False, "Rejected (Grid)", meta
        found_tokens = [t for t in self.ocr_circuit_tokens if t in ocr_text]
        meta["gates"] = list(set(found_tokens))
        for algo in self.algo_keywords:
            if algo in caption_text.lower(): meta["algorithm"] = algo.title(); break
        if len(meta["gates"]) >= 1: return True, "Valid (Gates Found)", meta
        return False, "Ambiguous", meta
