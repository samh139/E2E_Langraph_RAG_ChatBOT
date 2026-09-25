import re
import uuid

from difflib import SequenceMatcher
from typing import Any, Dict, List


class ChunkingEngine:

    def __init__(
        self,
        max_chars: int = 2500,
        overlap: int = 250,
        max_heading_levels: int = 4,
        min_heading_len: int = 3,
    ):
        self.max_chars = max_chars
        self.overlap = overlap
        self.max_heading_levels = max_heading_levels
        self.min_heading_len = min_heading_len

        if self.max_chars <= 0:
            raise ValueError(
                "max_chars must be greater than 0"
            )

        if self.overlap < 0:
            self.overlap = 0

        if self.overlap >= self.max_chars:
            self.overlap = int(
                self.max_chars * 0.1
            )

    # =========================================================
    # NORMALIZATION
    # =========================================================

    @staticmethod
    def normalize_text(text: str) -> str:

        if not text:
            return ""

        # Preserve the useful behavior from the original code:
        # fix words broken across lines.
        text = re.sub(
            r"-\n(\w+)",
            r"\1",
            text
        )

        text = text.replace(
            "\r",
            "\n"
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        return text.strip()

    @staticmethod
    def normalize_for_dedup(text: str) -> str:

        if not text:
            return ""

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        text = re.sub(
            r"[^\w\s]",
            "",
            text
        )

        return text.strip()

    # =========================================================
    # SENTENCE SPLITTING
    # =========================================================

    @staticmethod
    def split_sentences(text: str) -> List[str]:

        if not text:
            return []

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text.strip()
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # =========================================================
    # HEADING DETECTION
    # =========================================================

    @staticmethod
    def is_regex_heading(text: str) -> bool:

        text = text.strip()

        if not text:
            return False

        # Chapter 1 Something
        if re.match(
            r"^Chapter\s+\d+\s+[^.!?]{3,120}$",
            text,
            re.IGNORECASE
        ):
            return True

        # 8.1 Identity Information
        # 10.1 Collect Only What Is Necessary
        if re.match(
            r"^\d+(?:\.\d+)+\s+[^.!?]{3,120}$",
            text
        ):
            return True

        # 8. Sensitive Data Handling
        if re.match(
            r"^\d+\.\s+[A-Z][^.!?]{3,120}$",
            text
        ):
            return True

        return False

    # =========================================================
    # PARAGRAPH HANDLING
    # =========================================================

    def split_text_preserve_paragraphs(
        self,
        text: str
    ) -> List[str]:

        paras = [
            p.strip()
            for p in re.split(
                r"\n\s*\n",
                text
            )
            if p.strip()
        ]

        if not paras:

            paras = [
                ln.strip()
                for ln in text.splitlines()
                if ln.strip()
            ]

        return paras if paras else [text]

    # =========================================================
    # SENTENCE-AWARE CHUNKING
    # =========================================================

    def split_long_paragraph(
        self,
        paragraph: str
    ) -> List[str]:

        if len(paragraph) <= self.max_chars:
            return [paragraph]

        sentences = self.split_sentences(
            paragraph
        )

        if not sentences:
            return self._hard_split(
                paragraph
            )

        chunks = []

        current = ""

        for sentence in sentences:

            candidate = (
                sentence
                if not current
                else current + " " + sentence
            )

            if len(candidate) <= self.max_chars:

                current = candidate

            else:

                if current:
                    chunks.append(
                        current
                    )

                if len(sentence) <= self.max_chars:

                    current = sentence

                else:

                    chunks.extend(
                        self._hard_split(
                            sentence
                        )
                    )

                    current = ""

        if current:
            chunks.append(current)

        return chunks

    # =========================================================
    # HARD SPLIT WITH OVERLAP
    # =========================================================

    def _hard_split(
        self,
        text: str
    ) -> List[str]:

        chunks = []

        start = 0

        while start < len(text):

            target_end = min(
                start + self.max_chars,
                len(text)
            )

            end = target_end

            if target_end < len(text):

                nearest_space = text.rfind(
                    " ",
                    start,
                    target_end
                )

                if nearest_space > start:

                    end = nearest_space

            part = text[
                start:end
            ].strip()

            if part:
                chunks.append(part)

            if end >= len(text):
                break

            next_start = end - self.overlap

            start = max(
                next_start,
                start + 1
            )

        return chunks

    # =========================================================
    # GENERIC WINDOWED SPLIT
    # =========================================================

    def windowed_split(
        self,
        text: str
    ) -> List[str]:

        if not text:
            return []

        paragraphs = (
            self.split_text_preserve_paragraphs(
                text
            )
        )

        chunks = []

        current = ""

        for paragraph in paragraphs:

            paragraph = self.normalize_text(
                paragraph
            )

            if not paragraph:
                continue

            # Paragraph itself is too large.
            if len(paragraph) > self.max_chars:

                if current:

                    chunks.append(
                        current.strip()
                    )

                    current = ""

                chunks.extend(
                    self.split_long_paragraph(
                        paragraph
                    )
                )

                continue

            candidate = (
                paragraph
                if not current
                else current
                + "\n\n"
                + paragraph
            )

            if len(candidate) <= self.max_chars:

                current = candidate

            else:

                if current:
                    chunks.append(
                        current.strip()
                    )

                current = paragraph

        if current:
            chunks.append(
                current.strip()
            )

        # -----------------------------------------------------
        # Apply overlap between chunks.
        #
        # This fixes an important weakness in the original
        # implementation: overlap previously only happened
        # during the final safety split.
        # -----------------------------------------------------

        if self.overlap <= 0:
            return chunks

        overlapped = []

        for index, chunk in enumerate(chunks):

            if index == 0:

                overlapped.append(chunk)
                continue

            previous = chunks[index - 1]

            overlap_text = self._get_overlap_text(
                previous
            )

            if overlap_text:

                candidate = (
                    overlap_text
                    + "\n\n"
                    + chunk
                )

                if len(candidate) <= self.max_chars:

                    overlapped.append(
                        candidate
                    )

                else:

                    # Keep the current chunk intact
                    # rather than exceeding max_chars.
                    overlapped.append(
                        chunk
                    )

            else:

                overlapped.append(
                    chunk
                )

        return overlapped

    def _get_overlap_text(
        self,
        text: str
    ) -> str:

        if not text:
            return ""

        if len(text) <= self.overlap:
            return text

        candidate = text[
            -self.overlap:
        ]

        # Prefer starting at a word boundary.
        first_space = candidate.find(" ")

        if first_space > 0:
            candidate = candidate[
                first_space + 1:
            ]

        return candidate.strip()

    # =========================================================
    # ELEMENT → SECTION BUILDING
    # =========================================================

    def build_sections(
        self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        sections = []

        current = {
            "heading": None,
            "subheading": None,
            "paragraphs": [],
            "pages": set(),
        }

        def flush_section(
            keep_heading: bool = True
        ):

            if not current["paragraphs"]:
                return

            body = "\n\n".join(
                current["paragraphs"]
            ).strip()

            if not body:
                return

            sections.append(
                {
                    "heading": current["heading"],
                    "subheading": current["subheading"],
                    "body": body,
                    "pages": sorted(
                        current["pages"]
                    ),
                }
            )

            current["paragraphs"].clear()
            current["pages"].clear()

            if not keep_heading:
                current["heading"] = None

            current["subheading"] = None

        for element in elements:

            text = self.normalize_text(
                element.get(
                    "text",
                    ""
                )
            )

            if not text:
                continue

            page = element.get(
                "page"
            )

            element_type = element.get(
                "element_type",
                "paragraph"
            )

            heading_level = element.get(
                "heading_level"
            )

            # -------------------------------------------------
            # Explicit heading from extractor
            # -------------------------------------------------

            if (
                element_type == "heading"
                or heading_level == 1
            ):

                flush_section(
                    keep_heading=False
                )

                current["heading"] = text

                if page is not None:
                    current["pages"].add(
                        page
                    )

                continue

            # -------------------------------------------------
            # Explicit subheading
            # -------------------------------------------------

            if (
                element_type == "subheading"
                or heading_level == 2
            ):

                flush_section(
                    keep_heading=True
                )

                current["subheading"] = text

                if page is not None:
                    current["pages"].add(
                        page
                    )

                continue

            # -------------------------------------------------
            # Regex-based heading fallback
            # -------------------------------------------------

            if self.is_regex_heading(text):

                flush_section(
                    keep_heading=False
                )

                current["heading"] = text

                if page is not None:
                    current["pages"].add(
                        page
                    )

                continue

            # -------------------------------------------------
            # Body
            # -------------------------------------------------

            if current["paragraphs"]:

                last = current[
                    "paragraphs"
                ][-1]

                # Preserve the useful line-merging behavior
                # from the original implementation.
                if re.search(
                    r'[.!?]"?$|\)$',
                    last.strip()
                ):

                    current[
                        "paragraphs"
                    ].append(text)

                else:

                    current[
                        "paragraphs"
                    ][-1] = (
                        last
                        + " "
                        + text
                    )

            else:

                current[
                    "paragraphs"
                ].append(text)

            if page is not None:

                current[
                    "pages"
                ].add(page)

        flush_section()

        return sections

    # =========================================================
    # SECTION → CHUNKS
    # =========================================================

    def create_chunks(
        self,
        sections: List[Dict[str, Any]],
        file_name: str
    ) -> List[Dict[str, Any]]:

        chunks = []

        for section in sections:

            heading = section.get(
                "heading"
            )

            subheading = section.get(
                "subheading"
            )

            body = section.get(
                "body",
                ""
            )

            pages = section.get(
                "pages",
                []
            )

            prefix_parts = []

            if heading:
                prefix_parts.append(
                    heading
                )

            if subheading:
                prefix_parts.append(
                    subheading
                )

            prefix = "\n\n".join(
                prefix_parts
            ).strip()

            full_text = (
                prefix
                + "\n\n"
                + body
                if prefix
                else body
            )

            split_parts = self.windowed_split(
                full_text
            )

            for part in split_parts:

                if not part.strip():
                    continue

                chunks.append(
                    {
                        "id": str(
                            uuid.uuid4()
                        ),

                        "file_name": file_name,

                        "pages": pages,

                        "heading": heading,

                        "subheading": subheading,

                        "content": part,

                        "char_count": len(
                            part
                        ),
                    }
                )

        return self.deduplicate_chunks(
            chunks
        )

    # =========================================================
    # CHUNK DEDUPLICATION
    # =========================================================

    def deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        if not chunks:
            return []

        unique_chunks = []

        exact_seen = set()

        for chunk in chunks:

            content_key = (
                self.normalize_for_dedup(
                    chunk.get(
                        "content",
                        ""
                    )
                )
            )

            heading_key = (
                self.normalize_for_dedup(
                    chunk.get(
                        "heading",
                        ""
                    ) or ""
                )
            )

            subheading_key = (
                self.normalize_for_dedup(
                    chunk.get(
                        "subheading",
                        ""
                    ) or ""
                )
            )

            exact_key = (
                heading_key,
                subheading_key,
                content_key
            )

            if exact_key in exact_seen:
                continue

            exact_seen.add(
                exact_key
            )

            is_duplicate = False

            for existing in unique_chunks:

                existing_heading = (
                    self.normalize_for_dedup(
                        existing.get(
                            "heading",
                            ""
                        ) or ""
                    )
                )

                existing_subheading = (
                    self.normalize_for_dedup(
                        existing.get(
                            "subheading",
                            ""
                        ) or ""
                    )
                )

                if (
                    heading_key
                    != existing_heading
                ):
                    continue

                if (
                    subheading_key
                    != existing_subheading
                ):
                    continue

                existing_content = (
                    self.normalize_for_dedup(
                        existing.get(
                            "content",
                            ""
                        )
                    )
                )

                if (
                    not content_key
                    or not existing_content
                ):
                    continue

                similarity = (
                    SequenceMatcher(
                        None,
                        content_key,
                        existing_content
                    ).ratio()
                )

                if similarity >= 0.85:

                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_chunks.append(
                    chunk
                )

        return unique_chunks

    # =========================================================
    # PUBLIC METHOD
    # =========================================================

    def chunk(
        self,
        elements: List[Dict[str, Any]],
        file_name: str
    ) -> List[Dict[str, Any]]:

        if not elements:
            return []

        sections = self.build_sections(
            elements
        )

        return self.create_chunks(
            sections,
            file_name
        )