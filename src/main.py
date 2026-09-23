import os, sys, json, csv, time, re, shutil, logging, requests
import fitz, cv2, torch, pytesseract
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import softmax

# Modular Imports
from config import CONFIG, BASE_DIR
from scanner import ReferenceScanner
from processor import CircuitProcessor, FigureExtractor
from validator import CircuitValidator

if CONFIG["TESSERACT_PATH"]:
    pytesseract.pytesseract.tesseract_cmd = CONFIG["TESSERACT_PATH"]
    print(f"System : start processing")
else:
    print("start processing")
for d in [CONFIG["OUTPUT_IMG_DIR"], CONFIG["LOG_DIR"], CONFIG["TEMP_DIR"]]:
    os.makedirs(d, exist_ok=True)
fitz.TOOLS.mupdf_display_errors(False)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def setup_clip_model(): 
    print("--- SYSTEM: Loading CLIP AI Model into Memory... ---")
    try:
        model_name = "openai/clip-vit-base-patch32" 
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        model.to(device) 
        print(f"--- SYSTEM: AI Ready on {device.upper()} ---")
        return model, processor, device
    except Exception as e:
        print(f"[CRITICAL WARNING] Could not load AI Model: {e}")
        return None, None, None

def perform_periodic_cleanup(model, processor, device, folder_path): 
    if model is None: return 0
    print(f"\nAnalyzing images in {folder_path}...")
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)]
    deleted_count = 0
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            image = Image.open(filepath).convert("RGB") 
            inputs = processor( 
                text=CONFIG["CLIP_PROMPTS"], 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            with torch.no_grad(): 
                outputs = model(**inputs)
            probs = softmax(outputs.logits_per_image, dim=1).cpu().numpy()[0] 
            circuit_prob = probs[0] 
            if circuit_prob < CONFIG["CLIP_THRESHOLD"]: 
                image.close() 
                os.remove(filepath)
                deleted_count += 1
        except Exception as e:
            print(f"Failed to check {filename}: {e}")
    if deleted_count > 0:
        print(f"Removed {deleted_count} Invalid images.")
    else:
        print("No Invalid Images found.")
    return deleted_count

def download_pdf(arxiv_id, save_dir): 
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    path = os.path.join(save_dir, f"{arxiv_id}.pdf")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
            return path
        else:
            print(f"Failed to download {arxiv_id} (Status: {response.status_code})")
            return None
    except Exception as e:
        print(f"Exception downloading {arxiv_id}: {e}")
        return None

def parse_arxiv_ids(file_path): 
    ids = []
    if not os.path.exists(file_path):
        print(f"Error: Paper list file not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        matches = re.findall(r'arXiv:(\d{4}\.\d{4,5})', content)
        seen = set()
        for m in matches:
            if m not in seen:
                seen.add(m)
                ids.append(m)
    return ids

def main(): 
    if os.path.exists(CONFIG["OUTPUT_IMG_DIR"]):
        print(f"Removing existing output directory")
        shutil.rmtree(CONFIG["OUTPUT_IMG_DIR"])
    os.makedirs(CONFIG["OUTPUT_IMG_DIR"], exist_ok=True)
    json_path = os.path.join(BASE_DIR, "data", "02_extracted_dataset", f"dataset_MASTER_{CONFIG['EXAM_ID']}.json")
    if os.path.exists(json_path):
        os.remove(json_path)
    csv_path = os.path.join(BASE_DIR, "data", "02_extracted_dataset", f"paper_list_counts_{CONFIG['EXAM_ID']}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    print("Starting fresh execution.\n")

    clip_model, clip_processor, clip_device = setup_clip_model()
    extractor = FigureExtractor(dpi=CONFIG["DPI"]) 
    validator = CircuitValidator()

    dataset = {} 
    csv_rows = []
    total_images_extracted = 0 
    stop_processing = False    
    
    arxiv_ids = parse_arxiv_ids(CONFIG['INPUT_FILE'])
    print(f"Found {len(arxiv_ids)} unique arXiv IDs.")

    for arxiv_id in arxiv_ids:
        if stop_processing: break 
        print(f"\nProcessing {arxiv_id}")
        pdf_path = download_pdf(arxiv_id, CONFIG["TEMP_DIR"])
        if not pdf_path:
            csv_rows.append([arxiv_id, 0])
            continue
        valid_in_paper = 0
        try: 
            doc = fitz.open(pdf_path)
            ref_scanner = ReferenceScanner(doc)
            for page in doc:
                if stop_processing: break 
                candidates = extractor.extract_candidates(page)
                for cand in candidates:
                    if stop_processing: break 
                    raw_img = extractor.render_figure(page, cand['rect'], cand['caption_rect'])
                    if raw_img is None or raw_img.width == 0 or raw_img.height == 0: continue
                    sub_imgs = CircuitProcessor.extract_sub_circuits(raw_img)
                    if not sub_imgs: sub_imgs = [raw_img]
                    for sub_idx, sub_img in enumerate(sub_imgs):
                        if stop_processing: break
                        is_valid, reason, meta = validator.validate(sub_img, cand['caption'])
                        if is_valid:
                            valid_in_paper += 1
                            total_images_extracted += 1 
                            match = re.search(r'(?:Fig\.?|Figure)\s*(\d+)', cand['caption'], re.IGNORECASE)
                            if match: fig_num = int(match.group(1))
                            else: continue
                            main_desc, main_pos = ref_scanner.get_caption_context(cand['caption'], fig_num)
                            other_refs, other_pos = ref_scanner.find_references_in_paper(fig_num)
                            all_descriptions = main_desc + other_refs
                            all_positions = main_pos + other_pos
                            if meta["algorithm"] == "unknown":
                                full_context_text = " ".join(all_descriptions).lower()
                                for algo in validator.algo_keywords:
                                    if algo in full_context_text:
                                        meta["algorithm"] = algo.title()
                                        break
                            part_label = chr(97+sub_idx) if len(sub_imgs) > 1 else "main"
                            suffix = f"_{part_label}" if len(sub_imgs) > 1 else ""
                            fname = f"{arxiv_id}_p{cand['page']}_img{fig_num}{suffix}.png"
                            sub_img.save(os.path.join(CONFIG["OUTPUT_IMG_DIR"], fname))
                            dataset[fname] = {
                                "arxiv_id": arxiv_id,
                                "page": cand['page'],
                                "figure_number": fig_num,
                                "part_label": part_label,
                                "quantum_gates": meta["gates"],
                                "quantum_problem": meta["algorithm"],
                                "descriptions": all_descriptions,
                                "text_positions": all_positions 
                            }
                            print(f"  Saved {fname} (Total: {total_images_extracted}/{CONFIG['TARGET_TOTAL_IMAGES']})")
                            if total_images_extracted > 0 and total_images_extracted % CONFIG["CLEANUP_INTERVAL"] == 0:
                                print(f"\n--- TRIGGERING CHECKPOINT AT {total_images_extracted} IMAGES ---")
                                if clip_model:
                                    perform_periodic_cleanup(clip_model, clip_processor, clip_device, CONFIG["OUTPUT_IMG_DIR"])
                                try:
                                    actual_files = [f for f in os.listdir(CONFIG["OUTPUT_IMG_DIR"]) if f.endswith(".png")]
                                    actual_count = len(actual_files)
                                    if total_images_extracted != actual_count:
                                        print(f"  [SYNC] Counter updated. Internal: {total_images_extracted} -> Disk: {actual_count}")
                                        total_images_extracted = actual_count
                                    else:
                                        print(f"  [SYNC] Verified: {actual_count} valid images secured.")
                                except Exception as sync_err:
                                    print(f"  [SYNC] Error verifying files: {sync_err}")
                            if total_images_extracted >= CONFIG["TARGET_TOTAL_IMAGES"]:
                                print("TARGET REACHED. STOPPING...")
                                stop_processing = True
                                break
            doc.close()
        except Exception as e:
            print(f"Error processing PDF content for {arxiv_id}: {e}")
        try:
            os.remove(pdf_path)
            print(f"Deleted temp file: {pdf_path}")
        except OSError as e:
            print(f"Error deleting file {pdf_path}: {e}")
        csv_rows.append([arxiv_id, valid_in_paper])
        if not stop_processing:
            print("Waiting 3 seconds before next download...")
            time.sleep(3)
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print("\n--- Performing Final File Count Verification ---")
    actual_saved_files = [f for f in os.listdir(CONFIG["OUTPUT_IMG_DIR"]) if f.endswith(".png")]
    final_counts = {row[0]: 0 for row in csv_rows} 
    total_verified_images = 0
    for filename in actual_saved_files:
        if "_p" in filename:
            pdf_id = filename.split("_p")[0]
            if pdf_id in final_counts:
                final_counts[pdf_id] += 1
                total_verified_images += 1
    final_csv_rows = []
    processed_ids_set = set()
    for row in csv_rows:
        pid = row[0]
        processed_ids_set.add(pid)
        count = final_counts.get(pid, 0)
        final_csv_rows.append([pid, count])
    for original_id in arxiv_ids:
        if original_id not in processed_ids_set:
            final_csv_rows.append([original_id, ""])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["arxiv_id", "images_found"])
        writer.writerows(final_csv_rows)
    print(f"--------------------------------------------------")
    print(f"Verification Complete.")
    print(f"Internal Counter said: {total_images_extracted}")
    print(f"ACTUAL Files on Disk:  {total_verified_images}")
    print(f"CSV updated at: {csv_path}")

if __name__ == "__main__":
    main()
