# Automated Quantum Circuit Extraction Pipeline

An automated, multi-stage data engineering pipeline designed to construct a large-scale, semantically enriched dataset of quantum circuit diagrams from unstructured arXiv PDF preprints. 

Standard extraction methods fail on scientific literature due to visual disambiguation (e.g., 2D plots masquerading as circuits) and vector fragmentation (diagrams stored as thousands of disconnected paths). This pipeline resolves these challenges using a **Zero-Shot Hybrid Architecture** that combines deterministic Computer Vision heuristics with multimodal LLM semantic verification.

## System Architecture

The pipeline processes documents through four strict isolation layers to minimize expensive GPU inference:

1. **Vector De-fragmentation (Ingestion):** Utilizes `PyMuPDF` to parse display lists. A custom distance-based clustering algorithm merges disconnected vector paths into coherent figure boundaries.
2. **The Quality Gate (Primary Filter):** A highly optimized OpenCV suite that rejects false positives using strict geometric rules (Probabilistic Hough Transforms, L-Shape Detectors, Color Space Analysis).
3. **Semantic Verification (The Backstop):** Surviving candidates are processed through OpenAI's `CLIP-ViT-B/32` model in a zero-shot configuration to ensure semantic alignment.
4. **Hierarchical Context Miner (Enrichment):** A dual-pass regex engine executes localized caption searches and global document span searches to map exact string indices linking the visual asset to its algorithmic text.

## Repository Structure
text
├── data/
│   ├── 01_raw_pdfs/          # Input arXiv PDF manifests
│   └── 02_extracted_dataset/ # Synchronized JSON metadata, CSV reports, and high-res PNGs
├── src/                      # Core pipeline modules
│   ├── config.py             # Global thresholds and path configurations
│   ├── main.py               # Asynchronous execution loop
│   ├── validator.py          # QualityGate heuristics
│   ├── processor.py          # Morphological dilation and segmentation
│   └── scanner.py            # Dual-pass ContextMiner
├── docs/

│   └── version_history.md    # Audit ledger for threshold modifications
└── tests/