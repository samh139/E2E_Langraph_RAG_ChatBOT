import zipfile


SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


def get_file_type(file_path: str) -> str:
    """
    Identify supported document types from file content, not the filename.
    """
    import magic

    mime_type = magic.from_file(str(file_path), mime=True)
    if mime_type in SUPPORTED_MIME_TYPES:
        return SUPPORTED_MIME_TYPES[mime_type]

    # DOCX is a ZIP container. Some libmagic databases report the generic ZIP
    # MIME type, so verify the OOXML document members before routing it.
    if mime_type in {"application/zip", "application/x-zip-compressed"} and zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as archive:
            members = set(archive.namelist())
        if "[Content_Types].xml" in members and "word/document.xml" in members:
            return "docx"

    raise ValueError(f"Unsupported document content type: {mime_type}")


def get_extension(file_path: str) -> str:
    return f".{get_file_type(file_path)}"
