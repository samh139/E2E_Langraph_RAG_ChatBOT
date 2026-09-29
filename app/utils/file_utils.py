from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
}


def get_file_type(file_path: str) -> str:
    """
    Determine file type based on extension.
    """

    extension = Path(file_path).suffix.lower()

    file_type = SUPPORTED_EXTENSIONS.get(extension)

    if not file_type:
        raise ValueError(
            f"Unsupported file extension: {extension}"
        )

    return file_type


def get_extension(file_path: str) -> str:
    return Path(file_path).suffix.lower()