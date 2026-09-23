# Automated Quantum Circuit Extraction Pipeline

An automated, multi-stage data engineering pipeline designed to construct a large-scale, semantically enriched dataset of quantum circuit diagrams from unstructured arXiv PDF preprints. 

Standard extraction methods fail on scientific literature due to visual disambiguation (e.g., 2D plots masquerading as circuits) and vector fragmentation (diagrams stored as thousands of disconnected paths). This pipeline resolves these challenges using a **Zero-Shot Hybrid Architecture** that combines deterministic Computer Vision heuristics with multimodal LLM semantic verification.

## System Architecture

The pipeline processes documents through four strict isolation layers to minimize expensive GPU inference:

1. **Vector De-fragmentation (Ingestion):** Utilizes `PyMuPDF` to parse display lists. A custom distance-based clustering algorithm merges disconnected vector paths into coherent figure boundaries, rendering candidates at 300 DPI.
2. **The Quality Gate (Primary Filter):** A highly optimized OpenCV suite that rejects up to 90% of false positives using strict geometric rules:
   - **Probabilistic Hough Transforms:** Validates the presence of long horizontal timelines (wires) characteristic of quantum logic.
   - **L-Shape Detectors:** Analyzes image borders for axis lines to aggressively reject experimental 2D plots.
   - **Color Space Analysis:** Converts to HSV to isolate the saturation channel, rejecting heatmaps and colored bar charts based on solidity thresholds.
3. **Semantic Verification (The Backstop):** Surviving candidates are batched and processed through OpenAI's `CLIP-ViT-B/32` model in a zero-shot configuration to ensure the image semantically matches a quantum schematic rather than a generic flowchart.
4. **Hierarchical Context Miner (Enrichment):** A dual-pass regex engine that executes a localized caption search and a global document span search (via `re.finditer`) to map exact string indices linking the visual asset to its corresponding algorithmic text (e.g., Shor's, VQE).

## Repository Structure

```text
├── data/
│   ├── 01_raw_pdfs/          # Input arXiv PDF manifests 
│   └── 02_extracted_dataset/ # Synchronized JSON metadata, CSV reports, and high-res circuit PNGs
├── src/                      # Core pipeline modules
│   ├── config.py             # Global thresholds, regex patterns, and path configurations
│   ├── main.py               # Asynchronous execution loop and state management
│   ├── validator.py          # QualityGate heuristics (Hough Transforms, Color Masking)
│   ├── processor.py          # Morphological dilation and sub-circuit segmentation
│   └── scanner.py            # Dual-pass ContextMiner for exact text span alignment
├── docs/                     
│   └── version_history.md    # Audit ledger for threshold modifications and logic shifts
└── tests/                    # Unit testing for CV masks and extraction boundaries
