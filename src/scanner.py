import re

class ReferenceScanner: 
    def __init__(self, doc):
        self.doc = doc
        self.page_texts = {p.number: p.get_text("text").replace('\n', ' ') for p in doc}

    def get_caption_context(self, full_caption, fig_num): 
        if not full_caption: return [], []
        sentences = re.split(r'(?<=[.!?])\s+', full_caption.strip()) 
        sentences = [s.strip() for s in sentences if s.strip()]
        target_idx = -1
        pattern = re.compile(rf'(?:Fig\.?|Figure)\s*{fig_num}\b', re.IGNORECASE) 
        for i, sent in enumerate(sentences):
            if pattern.search(sent):
                target_idx = i
                break
        if target_idx == -1: 
            return [full_caption], [(0, len(full_caption))] 
        context = []
        if target_idx > 0: context.append(sentences[target_idx - 1]) 
        context.append(sentences[target_idx]) 
        if target_idx < len(sentences) - 1: context.append(sentences[target_idx + 1]) 
        if target_idx < len(sentences) - 2: context.append(sentences[target_idx + 2]) 
        final_text = " ".join(context)
        return [final_text], [(0, len(final_text))]
    
    def find_references_in_paper(self, fig_num): 
        references = []
        positions = []
        pattern_str = r'([^.!?]*?(?:Fig\.?|Figure)\s*' + str(fig_num) + r'\b[^.!?]*?[.!?])' 
        for page_num, text in self.page_texts.items():
            if len(text) < 50: continue 
            for match in re.finditer(pattern_str, text, re.IGNORECASE):
                sentence = match.group(0).strip()
                if len(sentence) > 20: 
                    clean_sent = re.sub(r'\s+', ' ', sentence)
                    references.append(f"[p{page_num+1}] {clean_sent}")
                    positions.append((match.start(), match.end()))
        return references, positions
