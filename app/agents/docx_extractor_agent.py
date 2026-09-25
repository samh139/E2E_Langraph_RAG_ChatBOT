from pathlib import Path
from docx import Document
from docx.text.paragraph import Paragraph
from app.workflow.state import IngestionState
from app.llm.llm_client import OllamaClient
from app.app_logger import LoggerFactory
from app.storage.storage_service import StorageService # Import your MongoDB StorageService

class DocxExtractionAgent:

    def __init__(self):
        # Decoupled state from initialization to match your updated orchestration schema
        self.ollama_cloud = OllamaClient()
        self.logger = LoggerFactory.get_logger(__name__)

    # MATCHES LANGGRAPH SIGNATURE: Accepts state and returns a state patch dictionary
    def execute(self, state: IngestionState) -> dict:
        # FIXED: Read values directly from Pydantic using dot-notation attributes
        file_path = state.file_path
        run_id = state.run_id
        self.logger.info(f"[DOCX Extractor] Processing: {file_path} | Run ID: {run_id}")

        document = Document(file_path)
        page_text = []   
        elements = []    

        paragraph_counter = 0
        image_counter = 0

        last_header = ""
        last_sub_header = ""

        # Ignore tiny bullet shapes, lines, and structural formatting symbols
        MIN_IMAGE_SIZE_BYTES = 1000 

        for child in document.element.body:
            
            if child.tag.endswith('p'):
                paragraph_counter += 1
                paragraph = Paragraph(child, document)
                text = paragraph.text.strip()

                style_name = paragraph.style.name if paragraph.style else ""
                style_lower = style_name.lower()

                element_type = "paragraph"
                heading_level = None

                if style_lower.startswith("heading 1"):
                    last_header = text
                    element_type = "heading"
                    heading_level = 1
                elif style_lower.startswith("heading 2"):
                    last_sub_header = text
                    element_type = "heading"
                    heading_level = 2
                elif style_lower.startswith("heading"):
                    element_type = "heading"
                    try:
                        heading_level = int(style_name.split()[-1])
                    except ValueError:
                        heading_level = None
                elif style_lower in {"title", "subtitle"}:
                    element_type = "heading"
                    heading_level = 1

                if text:
                    page_text.append(text)
                    elements.append({
                        "text": text,
                        "element_type": element_type,
                        "heading_level": heading_level,
                        "page": None, 
                        "style": style_name,
                        "paragraph_index": paragraph_counter,
                    })

                # --- Namespace-Agnostic Wildcard Extraction ---
                blips = child.xpath(".//*[local-name()='blip']")
                
                for blip in blips:
                    # Fetching embed code safely checking both potential URI attribute targets
                    rId = blip.get("{http://openxmlformats.org}embed") or blip.get("embed")
                    if not rId:
                        # Fallback query lookup attributes if attributes don't match structural standards namespace
                        for attr_name, attr_val in blip.attrib.items():
                            if attr_name.endswith('embed'):
                                rId = attr_val
                                break
                    
                    if not rId:
                        continue
                        
                    image_part = document.part.related_parts.get(rId)
                    if image_part is None:
                        continue

                    image_bytes = image_part.blob

                    if len(image_bytes) < MIN_IMAGE_SIZE_BYTES:
                        continue

                    image_counter += 1

                    # Fetch the contextual history chunk directly preceding the visual element
                    surrounding_context = "\n".join(page_text[-3:]).strip()

                    prompt = (
                        "You are an expert document analysis assistant. Transcribe this embedded document image or diagram precisely.\n\n"
                        f"DOCUMENT CONTEXT LOCATION HINT:\n"
                        f"Section: {last_header if last_header else 'Root'}\n"
                        f"Sub-Section: {last_sub_header if last_sub_header else 'None'}\n\n"
                        f"The text right above this image discusses:\n"
                        f"--- BEGIN CONTEXT ---\n"
                        f"{surrounding_context if surrounding_context else '[No paragraph text above this image]'}\n"
                        f"--- END CONTEXT ---\n\n"
                        "INSTRUCTIONS:\n"
                        "Using the layout location context above, analyze the chart, logic diagram, or code screenshot. "
                        "Precisely transcribe all syntax or structural code text into Markdown notation."
                    )

                    try:
                        self.logger.info(f"[DOCX Extractor] 🚀 Processing Image #{image_counter}. Triggering Ollama local inference...")
                        visual_description = self.ollama_cloud.analyze_vision(
                            image_bytes=image_bytes,
                            user_prompt=prompt,
                        )
                    except Exception as e:
                        self.logger.error(f"[DOCX Extractor Error] Visual parsing break on rId {rId}: {e}")
                        visual_description = ""

                    if visual_description and visual_description.strip():
                        visual_element_text = f"\n[IMAGE ANALYSIS: {visual_description.strip()}]\n"
                        page_text.append(visual_element_text)
                        elements.append({
                            "text": visual_element_text,
                            "element_type": "paragraph",
                            "page": None,
                            "style": "Ollama-Vision-Transcription",
                            "paragraph_index": paragraph_counter,
                        })

        final_text = "\n\n".join(page_text)
        self.logger.info(f"[DOCX Extractor] Finished. Total Images Processed={image_counter}")

        extraction_metadata = {
            "paragraph_count": paragraph_counter,
            "image_count": image_counter,
            "elements": elements,
        }

        # Safe local DB saving mechanism
        try:
            # Construct dictionary payload manually to mirror your other extractors
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
            mongo_id = StorageService.save_docx_data(mongo_payload, run_id=run_id)
            self.logger.info(f"[DOCX Extractor] Successfully persisted data to MongoDB. Doc Reference ID: {mongo_id}")
        except Exception as storage_err:
            self.logger.error(f"[DOCX Extractor Storage Error] Failed to write out to local database: {storage_err}")

        # CORRECT: Returns a clean patch dictionary for LangGraph to merge into state automatically
        return {
            "extracted_text": final_text,
            "extraction_metadata": extraction_metadata,
            "status": "extracted",
            "error": None
        }
