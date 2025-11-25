"""PDF document chunking."""

import logging
from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker

from mampfsearch.utils.models import Passage, FileLocation

logger = logging.getLogger(__name__)


def chunk_pdf_file(
    pdf_file_path: Path,
    enable_formula_enrichment: bool = False,
    max_tokens: int = 312,
) -> List[Passage]:
    """
    Extract and chunk text from a PDF file using Docling.

    Args:
        pdf_file_path: Path to the PDF file
        enable_formula_enrichment: If True, apply formula enrichment (experimental)

    Returns:
        List of Passage objects with FileLocation metadata
    """
    logger.info(f"Chunking PDF file: {pdf_file_path.name}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_formula_enrichment = enable_formula_enrichment

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(str(pdf_file_path))
    doc = result.document

    chunker = HybridChunker()
    chunker.tokenizer.max_tokens = max_tokens

    chunk_iter = chunker.chunk(dl_doc=doc)

    chunks = [Passage(text=chunk.text) for chunk in chunk_iter]

    logger.info(f"Extracted {len(chunks)} chunks from PDF")
    return chunks
