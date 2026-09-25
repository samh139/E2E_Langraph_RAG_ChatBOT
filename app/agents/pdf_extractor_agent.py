from pathlib import Path
import pymupdf
from datetime import datetime, timezone
from app.workflow.state import IngestionState
from app.llm.llm_client import OllamaClient
from app.app_logger import LoggerFactory
from app.storage.storage_service import StorageService 


class PDFExtractorAgent:

    def __init__(self):
        self.ollama_cloud = OllamaClient()
        self.logger = LoggerFactory.get_logger(__name__)

    # FIXED: Return type hint set to dict to match the state patch layout contract
    def execute(self, state: IngestionState) -> dict:
        file_path = state.file_path
        run_id = state.run_id 
        self.logger.info(f"[PDF Extractor] Processing: {file_path} | Run ID: {run_id}")

        extracted_text = []  
        page_metadata = []   
        elements = []        

        MIN_IMAGE_WIDTH = 100   
        MIN_IMAGE_HEIGHT = 100  

        with pymupdf.open(file_path) as document:

            # ------------------------------------------------------------------
            # Pass 1: collect every font size across the whole document so we
            # can compute a relative threshold for heading detection.
            # We do this before the main extraction loop so the threshold is
            # document-wide, not page-local (avoids false positives on pages
            # that happen to have unusually large body text).
            # ------------------------------------------------------------------
            all_font_sizes = []
            for page in document:
                for block in page.get_text("dict").get("blocks", []):
                    if block.get("type", 0) != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            size = span.get("size", 0)
                            if size > 0:
                                all_font_sizes.append(size)

            # Body font = the most common font size (mode approximation via median).
            # Heading threshold = body font × 1.2  (20% larger than typical body text).
            # Falls back to 14pt if the document has no font size information.
            if all_font_sizes:
                all_font_sizes_sorted = sorted(all_font_sizes)
                body_font_size = all_font_sizes_sorted[len(all_font_sizes_sorted) // 2]
                heading_font_threshold = body_font_size * 1.2
            else:
                body_font_size = 11.0
                heading_font_threshold = 14.0

            self.logger.info(
                f"[PDF Extractor] Font analysis: body={body_font_size:.1f}pt "
                f"heading_threshold={heading_font_threshold:.1f}pt"
            )

            # ------------------------------------------------------------------
            # Pass 2: main extraction loop
            # ------------------------------------------------------------------
            for page_number, page in enumerate(document):
                page_number_one_based = page_number + 1
                page_has_images = len(page.get_images(full=True)) > 0
                page_dict = page.get_text("dict")
                page_text = []  

                for block in page_dict.get("blocks", []):
                    block_type = block.get("type", 0)

                    # --------------------------------------------------------
                    # Image Block Processing (Type 1)
                    # --------------------------------------------------------
                    if block_type == 1:
                        if not page_has_images:
                            continue
                            
                        bbox = block.get("bbox")
                        if not bbox or len(bbox) < 4:
                            continue

                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]

                        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                            continue

                        try:
                            pix = page.get_pixmap(clip=bbox)
                            image_bytes = pix.tobytes("png")
                            surrounding_context = "\n".join(page_text).strip()
                            
                            prompt = (
                                "You are an expert document analysis assistant. Transcribe this embedded document image or diagram precisely.\n\n"
                                f"DOCUMENT CONTEXT HINT:\n"
                                f"The surrounding text on this page discusses the following topics:\n"
                                f"--- BEGIN CONTEXT ---\n"
                                f"{surrounding_context if surrounding_context else '[No text context available above this image]'}\n"
                                f"--- END CONTEXT ---\n\n"
                                "INSTRUCTIONS:\n"
                                "Using the context above, analyze the chart, diagram, or image. Extract all text, labels, data structures, and relationships. "
                                "Translate visual components into meaningful layout text that fits perfectly into the flow of the document content."
                            )

                            self.logger.info(f"[PDF Extractor] 🚀 Running Vision LLM on Page {page_number_one_based} ({int(width)}x{int(height)}px image)")
                            visual_description = self.ollama_cloud.analyze_vision(
                                image_bytes=image_bytes,
                                user_prompt=prompt,
                            )

                            if visual_description and visual_description.strip():
                                visual_element_text = f"\n[IMAGE ANALYSIS: {visual_description.strip()}]\n"
                                page_text.append(visual_element_text)
                                elements.append({
                                    "text": visual_element_text,
                                    "element_type": "paragraph",
                                    "page": page_number_one_based,
                                    "font_size": 11,
                                    "font": "Ollama-Cloud-Qwen-Vision",
                                })
                        except Exception as e:
                            self.logger.error(f"[PDF Extractor Error] Visual parsing break on page {page_number_one_based}: {e}")
                        continue

                    # --------------------------------------------------------
                    # Text Block Processing (Type 0)
                    # --------------------------------------------------------
                    if block_type == 0:
                        for line in block.get("lines", []):
                            line_text = ""
                            max_font_size = 0
                            font_names = set() 

                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if not text:
                                    continue

                                line_text += (" " if line_text else "") + text
                                max_font_size = max(max_font_size, span.get("size", 0))
                                font_names.add(span.get("font", ""))

                            line_text = line_text.strip()
                            if not line_text:
                                continue

                            page_text.append(line_text)

                            # --------------------------------------------------
                            # Heading promotion by font size.
                            #
                            # A line qualifies as a heading when ALL of:
                            #   1. Its max font size >= heading_font_threshold
                            #      (at least 20% larger than the document body font)
                            #   2. It is short enough to be a title — headings
                            #      rarely exceed 200 characters
                            #   3. It does not end with sentence-terminating
                            #      punctuation (heuristic: real headings don't
                            #      end with a full stop or comma)
                            #
                            # Font size 1.35× body → heading level 1 (major)
                            # Font size 1.20× body → heading level 2 (minor)
                            # --------------------------------------------------
                            element_type = "paragraph"
                            heading_level = None

                            is_short = len(line_text) <= 200
                            no_sentence_end = not line_text.rstrip().endswith(
                                (",", ".", ";", ":", "?", "!")
                            )

                            if max_font_size >= heading_font_threshold and is_short and no_sentence_end:
                                if max_font_size >= body_font_size * 1.35:
                                    element_type = "heading"
                                    heading_level = 1
                                else:
                                    element_type = "heading"
                                    heading_level = 2

                            elements.append({
                                "text": line_text,
                                "element_type": element_type,
                                "heading_level": heading_level,
                                "page": page_number_one_based,
                                "font_size": max_font_size,
                                "font": ", ".join(sorted(font_names)),
                            })

                page_raw_string = "\n".join(page_text)
                extracted_text.append(page_raw_string)

                page_metadata.append({
                    "page": page_number_one_based,
                    "characters": len(page_raw_string),
                })

        final_text = "\n\n".join(extracted_text)
        self.logger.info(f"[PDF Extractor] Done. Pages={len(page_metadata)} | Elements={len(elements)}")

        # Build metadata dictionary patch structure
        extraction_metadata = {
            "page_count": len(page_metadata),
            "pages": page_metadata,
            "elements": elements,
        }

        # Safe local DB saving mechanism
        try:
            # Reconstruct dictionary structure manually to keep MongoDB saves working cleanly
            mongo_payload = {
                "run_id": run_id,
                "file_path": file_path,
                "file_name": state.file_name,
                "file_extension": state.file_extension,
                "mime_type": state.mime_type,
                "file_type": state.file_type,
                "extracted_text": final_text,
                "extraction_metadata": extraction_metadata,
                "status": "extracted",
                "error": None
            }
            mongo_id = StorageService.save_pdf_data(mongo_payload, run_id=run_id)
            self.logger.info(f"[PDF Extractor] Successfully persisted data to MongoDB. Doc Reference ID: {mongo_id}")
        except Exception as storage_err:
            self.logger.error(f"[PDF Extractor Storage Error] Failed to write out to local database: {storage_err}")

        # CORRECT: Returns a clean state patch dictionary for LangGraph to merge automatically
        return {
            "run_id": run_id,
            "extracted_text": final_text,
            "extraction_metadata": extraction_metadata,
            "status": "extracted",
            "error": None
        }
