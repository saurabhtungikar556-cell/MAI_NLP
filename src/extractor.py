import fitz
import numpy as np

class FigureExtractor:
    def __init__(self, dpi=200):
        self.dpi = dpi

    def get_figures_from_page(self, page):
        """
        Extracts candidate figures by clustering vector drawings and images.
        Returns a list of dictionaries containing the crop and metadata.
        """
        candidates = []
        
        # 1. Get Vector Graphics (Drawings)
        paths = page.get_drawings()
        
        # Cluster paths that are close to each other to form "Figure Blocks"
        # Logic: If two paths are within 50px, they belong to the same figure.
        clusters = self._cluster_paths(paths, threshold=50)
        
        # 2. Get Raster Images (embedded photos)
        images = page.get_images(full=True)
        
        # Process Clusters (Potential Vector Figures)
        for cluster in clusters:
            rect = cluster['rect']
            # Reject tiny noise (e.g., page numbers, lines)
            if rect.width < 100 or rect.height < 100: 
                continue
                
            # Expand rect slightly to catch labels
            rect.x0 -= 10; rect.y0 -= 10; rect.x1 += 10; rect.y1 += 10
            
            # Find associated caption (look strictly below the rect)
            caption_text, caption_rect = self._find_caption(page, rect)
            
            candidates.append({
                "type": "vector",
                "rect": rect,
                "caption": caption_text,
                "caption_pos": caption_rect,
                "page": page.number + 1
            })

        # Process Raster Images
        for img_info in images:
            xref = img_info[0]
            img_rect = page.get_image_bbox(img_info)
            
            # Filter tiny icons/logos
            if img_rect.width < 100 or img_rect.height < 100:
                continue

            caption_text, caption_rect = self._find_caption(page, img_rect)
            
            candidates.append({
                "type": "raster",
                "rect": img_rect,
                "xref": xref,
                "caption": caption_text,
                "caption_pos": caption_rect,
                "page": page.number + 1
            })
            
        return candidates

    def _cluster_paths(self, paths, threshold):
        # Efficient clustering of bounding boxes (simplified logic for robustness)
        clusters = []
        for p in paths:
            r = p["rect"]
            merged = False
            for c in clusters:
                # Check intersection or proximity
                if r.intersects(c['rect']) or self._dist(r, c['rect']) < threshold:
                    c['rect'].include_rect(r) # Expand cluster
                    c['paths'].append(p)
                    merged = True
                    break
            if not merged:
                clusters.append({'rect': r, 'paths': [p]})
        return clusters

    def _dist(self, r1, r2):
        # Euclidean distance between two rectangles
        x_dist = max(r1.x0 - r2.x1, r2.x0 - r1.x1, 0)
        y_dist = max(r1.y0 - r2.y1, r2.y0 - r1.y1, 0)
        return (x_dist**2 + y_dist**2)**0.5

    def _find_caption(self, page, fig_rect):
        # Look for text starting with "Fig" or "Figure" strictly below the figure
        text_blocks = page.get_text("blocks")
        # Sort blocks by vertical position
        text_blocks.sort(key=lambda b: b[1]) 
        
        for b in text_blocks:
            # b = (x0, y0, x1, y1, text, ...)
            # Logic: Block must be below figure, close to it, and contain "Figure"
            if b[1] > fig_rect.y1 and (b[1] - fig_rect.y1) < 150: # within 150 units down
                text = b[4].strip().replace('\n', ' ')
                if text.lower().startswith(('fig', 'figure')):
                    return text, fitz.Rect(b[:4])
        return "", None

    def render_crop(self, page, rect):
        # High-quality render of the region
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect) # 2x zoom for clarity
        return pix