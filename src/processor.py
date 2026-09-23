import cv2, math, logging, fitz
import numpy as np
from PIL import Image
from config import CONFIG

class CircuitProcessor: 
    @staticmethod
    def extract_sub_circuits(pil_image): 
        img_np = np.array(pil_image)
        if len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR) 
        elif img_np.shape[2] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) 
        counts = np.bincount(gray.flatten())
        bg_color = np.argmax(counts) 
        if bg_color > 127: _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) 
        else: _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morphed = cv2.dilate(binary, kernel, iterations=2) 
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        if not contours: return []
        boxes = [] 
        page_area = img_np.shape[0] * img_np.shape[1]
        min_area = page_area * 0.005 
        for c in contours:
            x, y, w, h = cv2.boundingRect(c) 
            if (w * h) > min_area:
                boxes.append([x, y, x+w, y+h])
        if not boxes: return []
        merged_boxes = CircuitProcessor._merge_nearby_boxes( 
            boxes, 
            x_thresh=CONFIG["SPLIT_X_DIST"], 
            y_thresh=CONFIG["SPLIT_Y_DIST"]
        )
        buffer_points = CONFIG["CAPTION_BUFFER_PX"] 
        pixel_buffer = int(buffer_points * (CONFIG["DPI"] / 72)) 
        h_img, w_img = img_np.shape[:2]
        trim_limit = h_img - pixel_buffer 
        final_images = []
        padding = 10
        merged_boxes.sort(key=lambda b: b[1]) 
        for box in merged_boxes:
            x1, y1, x2, y2 = box
            x1 = max(0, x1 - padding); y1 = max(0, y1 - padding) 
            x2 = min(w_img, x2 + padding); y2 = min(h_img, y2 + padding)
            if y2 > trim_limit: 
                y2 = trim_limit
            if y2 <= y1: continue
            crop = img_np[y1:y2, x1:x2]
            final_images.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))) 
        return final_images
    
    @staticmethod
    def _merge_nearby_boxes(boxes, x_thresh, y_thresh): 
        while True:
            merged = False
            new_boxes = []
            used = [False] * len(boxes)
            for i in range(len(boxes)):
                if used[i]: continue
                current_box = list(boxes[i])
                used[i] = True
                for j in range(i + 1, len(boxes)):
                    if used[j]: continue
                    if CircuitProcessor._is_close(current_box, boxes[j], x_thresh, y_thresh): 
                        current_box[0] = min(current_box[0], boxes[j][0]) 
                        current_box[1] = min(current_box[1], boxes[j][1]) 
                        current_box[2] = max(current_box[2], boxes[j][2])
                        current_box[3] = max(current_box[3], boxes[j][3]) 
                        used[j] = True
                        merged = True
                new_boxes.append(current_box)
            boxes = new_boxes
            if not merged: break 
        return boxes
    
    @staticmethod
    def _is_close(box1, box2, x_thresh, y_thresh): 
        x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
        y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
        if x_overlap > 0: x_dist = 0 
        else: x_dist = max(box1[0] - box2[2], box2[0] - box1[2])
        if y_overlap > 0: y_dist = 0 
        else: y_dist = max(box1[1] - box2[3], box2[1] - box1[3])
        return x_dist < x_thresh and y_dist < y_thresh

class FigureExtractor: 
    def __init__(self, dpi=200):
        self.dpi = dpi
        
    def extract_candidates(self, page): 
        candidates = []
        page_num = page.number + 1
        paths = page.get_drawings() 
        clusters = self._cluster_paths(paths, threshold=CONFIG["CLUSTERING_THRESHOLD"]) 
        for cluster in clusters:
            rect = cluster['rect']
            if rect.width < 50 or rect.height < 50: continue 
            if rect.width > page.rect.width * 0.95 and rect.height > page.rect.height * 0.95: continue 
            caption_text, caption_rect = self._find_caption_smart(page, rect)
            candidates.append({"type": "vector", "rect": rect, "caption": caption_text, "caption_rect": caption_rect, "page": page_num})
        images = page.get_images(full=True)
        for img_info in images:
            try:
                rects = page.get_image_rects(img_info) 
                if not rects: continue
                img_rect = rects[0]
                if img_rect.width < 60 or img_rect.height < 60: continue
                caption_text, caption_rect = self._find_caption_smart(page, img_rect)
                candidates.append({"type": "raster", "rect": img_rect, "caption": caption_text, "caption_rect": caption_rect, "page": page_num})
            except Exception as e:
                logging.warning(f"Raster extraction failed on p{page_num}: {e}")
        return candidates
        
    def _cluster_paths(self, paths, threshold): 
        clusters = []
        for p in paths:
            r = p["rect"]
            if r.width < 1 and r.height < 1: continue 
            merged = False
            for c in clusters:
                dist = self._rect_distance(r, c['rect'])
                if dist < threshold:
                    c['rect'].include_rect(r) 
                    c['paths'].append(p)
                    merged = True
                    break
            if not merged: clusters.append({'rect': r, 'paths': [p]})
        return clusters
        
    def _rect_distance(self, r1, r2): 
        x_dist = max(r1.x0 - r2.x1, r2.x0 - r1.x1, 0)
        y_dist = max(r1.y0 - r2.y1, r2.y0 - r1.y1, 0)
        return math.sqrt(x_dist**2 + y_dist**2)
        
    def _find_caption_smart(self, page, fig_rect): 
        text_blocks = page.get_text("blocks")
        candidates = []
        search_area = fitz.Rect(fig_rect.x0 - 50, fig_rect.y0 - 50, fig_rect.x1 + 50, fig_rect.y1 + 200) 
        for b in text_blocks:
            b_rect = fitz.Rect(b[:4])
            text = b[4].strip().replace('\n', ' ')
            if b_rect.intersects(search_area):
                y_dist = b_rect.y0 - fig_rect.y1 
                if 0 <= y_dist < 150: candidates.append((b_rect, text, y_dist)) 
        candidates.sort(key=lambda x: x[2]) 
        for rect, text, dist in candidates:
            if text.lower().startswith(("fig", "figure")): return text, rect
        if candidates and len(candidates[0][1]) > 20: return candidates[0][1], candidates[0][0] 
        return "", None
        
    def render_figure(self, page, fig_rect, cap_rect): 
        if cap_rect:
            trimmed_cap = fitz.Rect(cap_rect)
            trimmed_cap.y1 = trimmed_cap.y0 + CONFIG["CAPTION_BUFFER_PX"] 
            final_rect = fig_rect | trimmed_cap 
        else:
            final_rect = fig_rect
        padding = 10
        final_rect.x0 -= padding; final_rect.y0 -= padding; final_rect.x1 += padding; final_rect.y1 += padding
        final_rect.intersect(page.rect) 
        pix = page.get_pixmap(matrix=fitz.Matrix(CONFIG["DPI"]/72, CONFIG["DPI"]/72), clip=final_rect) 
        try: return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except ValueError: return Image.frombytes("RGB", [pix.width, pix.height], pix.samples_mv)
