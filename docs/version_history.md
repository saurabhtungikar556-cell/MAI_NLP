# Pipeline Version History & Audit Ledger

This document tracks all mathematical thresholds, logic modifications, and architectural changes across the quantum circuit extraction pipeline.

## [v1.0.0] - 2026-09-23

**Architectural Baseline**

### System Changes

- Decoupled monolithic extraction script into modular architecture (`config.py`, `main.py`, `processor.py`, `validator.py`, `scanner.py`).
- Established strict data boundaries, ignoring raw PDFs and generated datasets from version control to prevent repository bloat.

### Baseline Calculations & Thresholds

- **Target Images:** `250`
- **CLIP Semantic Threshold:** `0.16` (ViT-B/32)
- **Vector Clustering Distance:** `50`
- **Caption Proximity Buffer:** `3.5` points
- **Color Block Solidity:** `> 500` contour area
- **Hough Transform Line Length:** `> 20%` of image width for horizontal wire validation

### Notes

- Initial integration test verified via arXiv ID `2502.01146`.
- Future modifications to OpenCV kernel dilations or PyTorch confidence scores must be logged as minor version increments (v1.1.0, etc.) below this entry.
