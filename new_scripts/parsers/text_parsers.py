import sys
import os
import time
import argparse

import pdfplumber
sys.path.append(os.getcwd())
import json
from abc import ABC, abstractmethod
from pathlib import Path
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from langchain_core.documents.base import Blob
from langchain_community.document_loaders.parsers import PyPDFium2Parser as LCParser

class BasePdfParser(ABC):
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.total_parse_time = 0.0
        self.files_processed = 0

    @abstractmethod
    def partition_pdf(self, file_path: Path):
        pass

    def _save_text(self, text_data: list, file_path: Path, lib_name: str) -> None:
        file_stem = file_path.stem
        result_path = self.results_dir / lib_name
        Path(result_path).mkdir(parents=True, exist_ok=True)
        result_file = result_path / f"{file_stem}_{lib_name}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(text_data, f, ensure_ascii=False, indent=4)

    def _get_difficulty(self, file_path: Path) -> str:
        """Extract difficulty level from file path (easy/medium/hard)."""
        path_parts = file_path.parts
        for part in path_parts:
            if part.lower() in ['easy', 'medium', 'hard']:
                return part.lower()
        return 'unknown'  # fallback if no difficulty found

    def _log_progress(self, file_path: Path, index: int, total: int) -> None:
        print(f"[{self.lib_name}] Processing {file_path.name} ({index}/{total})")

    def get_timing_summary(self) -> str:
        """Return a formatted string with timing information."""
        avg_time = self.total_parse_time / self.files_processed if self.files_processed > 0 else 0
        return (f"Parser: {self.lib_name} | "
                f"Total time: {self.total_parse_time:.2f}s | "
                f"Files: {self.files_processed} | "
                f"Avg per file: {avg_time:.3f}s")


class PyPdf2Parser(BasePdfParser):
    def __init__(self, results_dir: Path) -> None:
        super().__init__(results_dir)
        self.lib_name = 'pypdf2'

    def partition_pdf(self, file_path: Path):
        start_time = time.time()
        try:
            stem = file_path.stem
            page_num = int(''.join(filter(str.isdigit, stem))[-2:] or '1')
            difficulty = self._get_difficulty(file_path)
            text_data = []
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                if reader.pages:
                    content = reader.pages[0].extract_text() or ""
                    text_data.append({
                        'page_number': page_num, 
                        'text': content.strip(),
                        'difficulty': difficulty
                    })
            self._save_text(text_data, file_path, self.lib_name)
        except Exception as e:
            print(f"Failed to parse {file_path} with {self.lib_name}: {e}")
        finally:
            parse_time = time.time() - start_time
            self.total_parse_time += parse_time
            self.files_processed += 1

class PyMuPdfParser(BasePdfParser):
    def __init__(self, results_dir: Path) -> None:
        super().__init__(results_dir)
        self.lib_name = 'pymupdf'

    def partition_pdf(self, file_path: Path):
        start_time = time.time()
        try:
            stem = file_path.stem
            page_num = int(''.join(filter(str.isdigit, stem))[-2:] or '1')
            difficulty = self._get_difficulty(file_path)
            text_data = []
            doc = fitz.open(file_path)
            if doc.page_count > 0:
                text = doc.load_page(0).get_text() or ""
                text_data.append({
                    'page_number': page_num, 
                    'text': text.strip(),
                    'difficulty': difficulty
                })
            doc.close()
            self._save_text(text_data, file_path, self.lib_name)
        except Exception as e:
            print(f"Failed to parse {file_path} with {self.lib_name}: {e}")
        finally:
            parse_time = time.time() - start_time
            self.total_parse_time += parse_time
            self.files_processed += 1

class PdfMinerParser(BasePdfParser):
    def __init__(self, results_dir: Path) -> None:
        super().__init__(results_dir)
        self.lib_name = 'pdfminer'

    def partition_pdf(self, file_path: Path):
        start_time = time.time()
        try:
            from pdfminer.pdfpage import PDFPage
            from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
            from pdfminer.converter import TextConverter
            from pdfminer.layout import LAParams
            from io import StringIO

            stem = file_path.stem
            page_num = int(''.join(filter(str.isdigit, stem))[-2:] or '1')
            difficulty = self._get_difficulty(file_path)
            text_data = []
            with open(file_path, 'rb') as f:
                for page in PDFPage.get_pages(f):
                    output = StringIO()
                    rm = PDFResourceManager()
                    device = TextConverter(rm, output, laparams=LAParams())
                    PDFPageInterpreter(rm, device).process_page(page)
                    page_text = output.getvalue() or ""
                    text_data.append({
                        'page_number': page_num, 
                        'text': page_text.strip(),
                        'difficulty': difficulty
                    })
                    device.close()
                    output.close()
            self._save_text(text_data, file_path, self.lib_name)
        except Exception as e:
            print(f"Failed to parse {file_path} with {self.lib_name}: {e}")
        finally:
            parse_time = time.time() - start_time
            self.total_parse_time += parse_time
            self.files_processed += 1

class PdfPlumberParser(BasePdfParser):
    def __init__(self, results_dir: Path) -> None:
        super().__init__(results_dir)
        self.lib_name = 'pdfplumber'

    def partition_pdf(self, file_path: Path):
        start_time = time.time()
        try:
            stem = file_path.stem
            page_num = int(''.join(filter(str.isdigit, stem))[-2:] or '1')
            difficulty = self._get_difficulty(file_path)
            text_data = []
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages, start=1):
                    self._log_progress(file_path, i, total_pages)
                    content = page.extract_text() or ""
                    text_data.append({
                        'page_number': page_num,
                        'text': content.strip(),
                        'difficulty': difficulty
                    })
            self._save_text(text_data, file_path, self.lib_name)
        except Exception as e:
            print(f"Failed to parse {file_path} with {self.lib_name}: {e}")
        finally:
            parse_time = time.time() - start_time
            self.total_parse_time += parse_time
            self.files_processed += 1

class PyPDFium2Parser(BasePdfParser):
    def __init__(self, results_dir: Path) -> None:
        super().__init__(results_dir)
        self.lib_name = 'pypdfium2'

    def partition_pdf(self, file_path: Path):
        start_time = time.time()
        try:
            # Extract page number from filename (expected format: <base>_page_<N>.pdf)
            stem = file_path.stem
            page_num = int(''.join(filter(str.isdigit, stem))[-2:] or '1')
            difficulty = self._get_difficulty(file_path)

            # Load the PDF via LangChain's Blob
            blob = Blob.from_path(str(file_path))

            # Instantiate the LangChain PyPDFium2 parser
            parser = LCParser(
                extract_images=False,
                mode="page",
                pages_delimiter="\n\x0c"
            )

            # Use lazy_parse but collect into a list for counting
            documents = list(parser.lazy_parse(blob))
            total = len(documents)

            text_data = []
            # Collect text per parsed page (should be one page for single-page PDFs)
            for idx, doc in enumerate(documents, start=1):
                self._log_progress(file_path, idx, total)
                content = doc.page_content or ""
                text_data.append({
                    'page_number': page_num,
                    'text': content.strip(),
                    'difficulty': difficulty
                })

            # Save JSON results
            self._save_text(text_data, file_path, self.lib_name)

        except Exception as e:
            print(f"Failed to parse {file_path} with {self.lib_name}: {e}")
        finally:
            parse_time = time.time() - start_time
            self.total_parse_time += parse_time
            self.files_processed += 1


class UnstructuredParser(BasePdfParser):
    def __init__(self, results_dir: Path, strategy: str) -> None:
        super().__init__(results_dir)
        self.lib_name = 'unstructured'
        self.strategy = strategy

    def partition_pdf(self, file_path: Path):
        start_time = time.time()
        try:
            stem = file_path.stem
            page_num = int(''.join(filter(str.isdigit, stem))[-2:] or '1')
            difficulty = self._get_difficulty(file_path)
            elements = partition_pdf(
                filename=str(file_path),
                strategy=self.strategy,
                hi_res_model_name='yolox',
                infer_table_structure=True,
                languages=['eng']
            )
            text = "\n\n".join(el.text for el in elements if el.text)
            # Create directories
            elements_dir = self.results_dir / self.lib_name / 'elements'
            elements_dir.mkdir(parents=True, exist_ok=True)
            
            text_data = []
            text_data.append({
                'page_number': page_num,
                'text': text.strip(),
                'difficulty': difficulty
            })
            self._save_text(text_data, file_path, self.lib_name)
            elements_to_json(
                elements=elements,
                filename=str(elements_dir / f"{file_path.stem}.json"),
                indent=2,
            )
        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")
        finally:
            parse_time = time.time() - start_time
            self.total_parse_time += parse_time
            self.files_processed += 1
        

def get_dataset_paths(dataset_name):
    """Get the appropriate paths for the specified dataset."""
    if dataset_name == "tablequest":
        pages_dir = Path("tablequest/sampled_pages_pdf")
        results_dir = Path("new_scripts/data/parsed_pages/tablequest")
    else:  # default to original dataset
        pages_dir = Path("new_scripts/data/financebench_extracted_pages_pdf")
        results_dir = Path("new_scripts/data/parsed_pages/financebench")
    
    return pages_dir, results_dir


if __name__ == '__main__':
    # Choose which dataset to parse ('default' or 'tablequest')
    dataset = 'financebench'  # Change this to 'default' for the original dataset
    
    # Get paths based on dataset choice
    pages_dir, results_dir = get_dataset_paths(dataset)
    
    print(f"📂 Using dataset: {dataset}")
    print(f"📂 Pages directory: {pages_dir}")
    print(f"📂 Results directory: {results_dir}")
    print()

    parsers = [
        PyPdf2Parser(results_dir),
        PyMuPdfParser(results_dir),
        PdfMinerParser(results_dir),
        UnstructuredParser(results_dir, strategy='hi_res'),
        PdfPlumberParser(results_dir),
        PyPDFium2Parser(results_dir),
    ]

    pdf_files = list(pages_dir.glob("**/*.pdf"))  # Use recursive glob for nested directories
    total_files = len(pdf_files)
    print(f"Found {total_files} PDF files to process\n")

    # Store timing results for final summary
    timing_results = []

    for parser in parsers:
        print(f"🚀 Starting {parser.lib_name} parser...")
        parser_start_time = time.time()
        
        for idx, pdf_file in enumerate(pdf_files, 1):
            parser._log_progress(pdf_file, idx, total_files)
            parser.partition_pdf(pdf_file)
        
        parser_total_time = time.time() - parser_start_time
        timing_summary = parser.get_timing_summary()
        
        print(f"✅ Finished {parser.lib_name} parser")
        print(f"📊 {timing_summary}")
        print(f"⏱️  Total parser time (including overhead): {parser_total_time:.2f}s\n")
        
        timing_results.append({
            'parser': parser.lib_name,
            'total_time': parser.total_parse_time,
            'files_processed': parser.files_processed,
            'avg_per_file': parser.total_parse_time / parser.files_processed if parser.files_processed > 0 else 0,
            'total_with_overhead': parser_total_time
        })

    # Print final timing summary
    print("=" * 80)
    print("📈 FINAL TIMING SUMMARY")
    print("=" * 80)
    
    # Sort by total parsing time
    timing_results.sort(key=lambda x: x['total_time'])
    
    for result in timing_results:
        print(f"{result['parser']:<15} | "
              f"Parse: {result['total_time']:>8.2f}s | "
              f"Total: {result['total_with_overhead']:>8.2f}s | "
              f"Files: {result['files_processed']:>3d} | "
              f"Avg: {result['avg_per_file']:>6.3f}s/file")
    
    print("=" * 80)
    if timing_results and timing_results[0]['total_time'] > 0:
        print(f"Fastest parser (pure parsing): {timing_results[0]['parser']} ({timing_results[0]['total_time']:.2f}s)")
        print(f"Slowest parser (pure parsing): {timing_results[-1]['parser']} ({timing_results[-1]['total_time']:.2f}s)")
        print(f"Speed difference: {timing_results[-1]['total_time'] / timing_results[0]['total_time']:.1f}x slower")
    else:
        print("No files were processed - no timing comparison available")
    
    # Save timing results to JSON file
    timing_file = Path("new_scripts/parsers") / f"timing_results_{dataset}.json"
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata to the timing results
    timing_data = {
        'dataset': dataset,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_files': total_files,
        'pages_dir': str(pages_dir),
        'results_dir': str(results_dir),
        'timing_results': timing_results
    }
    
    with open(timing_file, 'w', encoding='utf-8') as f:
        json.dump(timing_data, f, indent=2)
    
    print(f"📊 Timing results saved to: {timing_file}")
