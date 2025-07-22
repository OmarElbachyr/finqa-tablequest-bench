import json
from typing import List
from chonkie import TokenChunker, SentenceChunker, SemanticChunker, RecursiveChunker, RecursiveRules, SDPMChunker, NeuralChunker
import tiktoken
import os
import glob

import torch


def apply_overlap_to_chunks(chunks: List[str], overlap_size: int, tokenizer) -> List[str]:
    """
    Apply overlap to chunks that don't natively support it.
    
    Args:
        chunks: List of text chunks
        overlap_size: Number of tokens to overlap between chunks
        tokenizer: Tokenizer to use for counting tokens
    
    Returns:
        List of chunks with overlap applied
    """
    if overlap_size <= 0 or len(chunks) <= 1:
        return chunks
    
    overlapped_chunks = []
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk remains unchanged
            overlapped_chunks.append(chunk)
        else:
            # Get tokens from previous chunk for overlap
            prev_chunk = chunks[i-1]
            prev_tokens = tokenizer.encode(prev_chunk)
            current_tokens = tokenizer.encode(chunk)
            
            # Take last 'overlap_size' tokens from previous chunk
            overlap_tokens = prev_tokens[-overlap_size:] if len(prev_tokens) > overlap_size else prev_tokens
            
            # Combine overlap with current chunk
            combined_tokens = overlap_tokens + current_tokens
            overlapped_text = tokenizer.decode(combined_tokens)
            
            overlapped_chunks.append(overlapped_text)
    
    return overlapped_chunks

def load_json_document(json_path):
    # Load the JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract the "text" field
    texts = [item["text"] for item in data]

    return texts[0]

def save_chunks_to_json(chunks: List[str], output_path: str) -> None:
    """
    Save a list of text chunks to a JSON file under key "chunks".

    Args:
        chunks: List of text chunk strings.
        output_path: Path where to write the JSON.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'chunks': chunks}, f, ensure_ascii=False, indent=2)
        
def token_chunker(text, tokenizer='gpt2', chunk_size=512, chunk_overlap=0, output_path=None): 
    tokenizer = tiktoken.get_encoding(tokenizer)

    chunker = TokenChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,  
        chunk_overlap=chunk_overlap,
        return_type="texts" 
    )
    
    chunks = chunker.chunk(text)

    if output_path:
        save_chunks_to_json(chunks, output_path)

    print(len(chunks), "chunks created with chunk size", chunk_size, "and overlap", chunk_overlap)
    return chunks

def sentence_chunker(text, tokenizer='gpt2', chunk_size=512, chunk_overlap=0, min_sentences_per_chunk=1, output_path=None):
    tokenizer = tiktoken.get_encoding(tokenizer)

    chunker = SentenceChunker(
        tokenizer_or_token_counter=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_sentences_per_chunk=min_sentences_per_chunk, 
        return_type="texts" 
    )

    chunks = chunker.chunk(text)


    if output_path:
        save_chunks_to_json(chunks, output_path)

    print(len(chunks), "chunks created with chunk size", chunk_size, "and overlap", chunk_overlap)
    return chunks

def semantic_chunker(text, tokenizer='gpt2', chunk_size=512, chunk_overlap=0, similarity_threshold=0.7, output_path=None):
    tokenizer_obj = tiktoken.get_encoding(tokenizer)

    chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-8M",  
        threshold=similarity_threshold,               
        chunk_size=chunk_size,                        
        min_sentences=1,
        return_type="texts"                              
    )

    chunks = chunker.chunk(text)
    
    # Apply overlap if specified, because this chunker doesn't support overlap natively 
    if chunk_overlap > 0:
        chunks = apply_overlap_to_chunks(chunks, chunk_overlap, tokenizer_obj)

    if output_path:
        save_chunks_to_json(chunks, output_path)

    print(len(chunks), "chunks created with chunk size", chunk_size, "and overlap", chunk_overlap, "and similarity threshold", similarity_threshold)
  
    return chunks

def sdpm_chunker(text, tokenizer='gpt2', chunk_size=512, chunk_overlap=0, similarity_threshold=0.7, output_path=None):
    tokenizer_obj = tiktoken.get_encoding(tokenizer)

    chunker = SDPMChunker(
        embedding_model="minishlab/potion-base-8M",  
        threshold=similarity_threshold,                              
        chunk_size=chunk_size,                            
        min_sentences=1,                           
        skip_window=1,                               
        return_type="texts",
    )

    chunks = chunker.chunk(text)
    
    # Apply overlap if specified, because this chunker doesn't support overlap natively 
    if chunk_overlap > 0:
        chunks = apply_overlap_to_chunks(chunks, chunk_overlap, tokenizer_obj)

    if output_path:
        save_chunks_to_json(chunks, output_path)

    print(len(chunks), "chunks created with chunk size", chunk_size, "and overlap", chunk_overlap, "and similarity threshold", similarity_threshold)
  
    return chunks

def recursive_chunker(text, tokenizer='gpt2', chunk_size=512, chunk_overlap=0, min_characters_per_chunk=24, output_path=None):
    tokenizer_obj = tiktoken.get_encoding(tokenizer)

    chunker = RecursiveChunker(
        tokenizer_or_token_counter=tokenizer_obj,
        chunk_size=chunk_size,
        min_characters_per_chunk=min_characters_per_chunk,
        rules=RecursiveRules(),
        return_type="texts"
    )

    chunks = chunker.chunk(text)

    # Apply overlap if specified, because this chunker doesn't support overlap natively 
    if chunk_overlap > 0:
        chunks = apply_overlap_to_chunks(chunks, chunk_overlap, tokenizer_obj)

    if output_path:
        save_chunks_to_json(chunks, output_path)

    print(len(chunks), "chunks created with chunk size", chunk_size, "and overlap", chunk_overlap, "and min_characters_per_chunk", min_characters_per_chunk)
  
    return chunks

def neural_chunker(text, tokenizer='gpt2', chunk_size=512, chunk_overlap=0, min_characters_per_chunk=24, output_path=None):
    tokenizer_obj = tiktoken.get_encoding(tokenizer)

    chunker = NeuralChunker(
        model="mirth/chonky_modernbert_base_1",  
        device_map="cuda" if torch.cuda.is_available() else "cpu",                         
        min_characters_per_chunk=10,             
        return_type="texts"                     
    )

    chunks = chunker.chunk(text)

    # Apply overlap if specified, because this chunker doesn't support overlap natively 
    if chunk_overlap > 0:
        chunks = apply_overlap_to_chunks(chunks, chunk_overlap, tokenizer_obj)

    if output_path:
        save_chunks_to_json(chunks, output_path)

    print(len(chunks), "chunks created with chunk size", chunk_size, "and overlap", chunk_overlap, "and min_characters_per_chunk", min_characters_per_chunk)
  
    return chunks

# def word_chunker(document, tokenizer='gpt2', mode='advanced', chunk_size=512, chunk_overlap=0):
#     """
#     mode: chunking mode
#         simple: basic space-based splitting
#         advanced: handles punctuation and special cases
#     """
#     tokenizer = AutoTikTokenizer.from_pretrained(tokenizer_name_or_path=tokenizer)
    
#     chunker = WordChunker(
#         tokenizer=tokenizer,
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         mode=mode
#     )

#     chonkie_chunks = chunker.chunk(document.page_content)
#     chunks = [Document(page_content=chunk.text, metadata={'source': document.metadata['source']}) for chunk in chonkie_chunks]

#     return chunks
     


def chunk_all_parsed_content(parsers, chunkers, input_base, output_base, chunk_size, overlap_sizes, dataset='financebench', SKIP_EXISTING=True):
    # Update paths to include dataset
    dataset_input_base = os.path.join(input_base, dataset)
    dataset_output_base = os.path.join(output_base, dataset)
    
    print(f"📂 Dataset: {dataset}")
    print(f"📂 Input base: {dataset_input_base}")
    print(f"📂 Output base: {dataset_output_base}")
    print()
    
    for parser in parsers:
        print(f"\n\n{'='*50}")
        print(f"Processing parser: {parser}")
        print('='*50)
        parser_path = os.path.join(dataset_input_base, parser)
        json_files = glob.glob(os.path.join(parser_path, '*.json'))
        total_files = len(json_files)

        for chunker in chunkers:
            print(f"\n{'-'*30}")
            print(f"Using chunking strategy: {chunker}")
            print('-'*30)
            
            for overlap_size in overlap_sizes:
                print(f"\n📊 Processing overlap size: {overlap_size}")
                print(f"Processing {total_files} files with overlap {overlap_size}...")
                
                for idx, json_file in enumerate(json_files, 1):
                    print(f"\nFile {idx}/{total_files}: {os.path.basename(json_file)}")
                    filename = os.path.splitext(os.path.basename(json_file))[0]
                    
                    if chunker == 'token':
                        out_folder = os.path.join(dataset_output_base, parser, 'token', f'overlap_{overlap_size}')
                        os.makedirs(out_folder, exist_ok=True)
                        out_path = os.path.join(out_folder, f"{filename}_chunks.json")
                        
                        # Check if file already exists and skip if SKIP_EXISTING is True
                        if SKIP_EXISTING and os.path.exists(out_path):
                            print(f"  ⏭️  Skipping {os.path.basename(out_path)} (already exists)")
                            continue
                        
                        text = load_json_document(json_file)
                        token_chunker(text, 'gpt2', chunk_size, overlap_size, out_path)
                    elif chunker == 'sentence':
                        out_folder = os.path.join(dataset_output_base, parser, 'sentence', f'overlap_{overlap_size}')
                        os.makedirs(out_folder, exist_ok=True)
                        out_path = os.path.join(out_folder, f"{filename}_chunks.json")
                        
                        # Check if file already exists and skip if SKIP_EXISTING is True
                        if SKIP_EXISTING and os.path.exists(out_path):
                            print(f"  ⏭️  Skipping {os.path.basename(out_path)} (already exists)")
                            continue
                        
                        text = load_json_document(json_file)
                        sentence_chunker(text, 'gpt2', chunk_size, overlap_size, 
                                         min_sentences_per_chunk=1, output_path=out_path)
                    elif chunker == 'semantic':
                        out_folder = os.path.join(dataset_output_base, parser, 'semantic', f'overlap_{overlap_size}')
                        os.makedirs(out_folder, exist_ok=True)
                        out_path = os.path.join(out_folder, f"{filename}_chunks.json")
                        
                        # Check if file already exists and skip if SKIP_EXISTING is True
                        if SKIP_EXISTING and os.path.exists(out_path):
                            print(f"  ⏭️  Skipping {os.path.basename(out_path)} (already exists)")
                            continue
                        
                        text = load_json_document(json_file)
                        semantic_chunker(text, 'gpt2', chunk_size, overlap_size, 
                                       similarity_threshold=0.7, output_path=out_path)
                    elif chunker == 'recursive':
                        out_folder = os.path.join(dataset_output_base, parser, 'recursive', f'overlap_{overlap_size}')
                        os.makedirs(out_folder, exist_ok=True)
                        out_path = os.path.join(out_folder, f"{filename}_chunks.json")
                        
                        # Check if file already exists and skip if SKIP_EXISTING is True
                        if SKIP_EXISTING and os.path.exists(out_path):
                            print(f"  ⏭️  Skipping {os.path.basename(out_path)} (already exists)")
                            continue
                        
                        text = load_json_document(json_file)
                        recursive_chunker(text, 'gpt2', chunk_size, overlap_size, 
                                        min_characters_per_chunk=24, output_path=out_path)
                    elif chunker == 'sdpm':
                        out_folder = os.path.join(dataset_output_base, parser, 'sdpm', f'overlap_{overlap_size}')
                        os.makedirs(out_folder, exist_ok=True)
                        out_path = os.path.join(out_folder, f"{filename}_chunks.json")
                        
                        # Check if file already exists and skip if SKIP_EXISTING is True
                        if SKIP_EXISTING and os.path.exists(out_path):
                            print(f"  ⏭️  Skipping {os.path.basename(out_path)} (already exists)")
                            continue
                        
                        text = load_json_document(json_file)
                        sdpm_chunker(text, 'gpt2', chunk_size, overlap_size, 
                                   similarity_threshold=0.7, output_path=out_path)
                    elif chunker == 'neural':
                        out_folder = os.path.join(dataset_output_base, parser, 'neural', f'overlap_{overlap_size}')
                        os.makedirs(out_folder, exist_ok=True)
                        out_path = os.path.join(out_folder, f"{filename}_chunks.json")
                        
                        # Check if file already exists and skip if SKIP_EXISTING is True
                        if SKIP_EXISTING and os.path.exists(out_path):
                            print(f"  ⏭️  Skipping {os.path.basename(out_path)} (already exists)")
                            continue
                        
                        text = load_json_document(json_file)
                        neural_chunker(text, 'gpt2', chunk_size, overlap_size, 
                                     min_characters_per_chunk=24, output_path=out_path)
                print(f"\nCompleted {chunker} chunking for {parser} with overlap {overlap_size}")
     

if __name__ == '__main__':
    # Dataset selection: 'financebench' or 'tablequest'
    dataset = 'tablequest'  # Change to 'financebench' for the original dataset
    
    parsers=['pdfminer', 'pymupdf', 'pypdf2', 'unstructured', 'pdfplumber', 'pypdfium2']  # 'pdfminer', 'pymupdf', 'pypdf2', 'unstructured', 'pdfplumber', 'pypdfium2'
    chunkers=['token', 'sentence', 'semantic', 'recursive', 'sdpm', 'neural'] # 'token', 'sentence', 'semantic', 'recursive', 'sdpm', 'neural'
    overlap_sizes = [0, 128, 256] #, 128, 256, 512] 
    SKIP_EXISTING = True  # Set to False to recreate all files

    input_base='new_scripts/data/parsed_pages'
    output_base='new_scripts/data/parsed_pages_chunks'

    print(f"\n{'*'*60}")
    print(f"Starting chunking process...")
    print(f"Dataset: {dataset}")
    print(f"Parsers: {parsers}")
    print(f"Chunkers: {chunkers}")
    print(f"Overlap sizes: {overlap_sizes}")
    print(f"Skip existing: {SKIP_EXISTING}")
    print('*'*60)
    
    chunk_all_parsed_content(parsers, chunkers, input_base, output_base, 
                             chunk_size=512, overlap_sizes=overlap_sizes, 
                             dataset=dataset, SKIP_EXISTING=SKIP_EXISTING)
    """
       ** For 'sentence' chunking, the `min_sentences_per_chunk` is set to 1.
       ** For 'semantic' chunking, the `similarity_threshold` is set to 0.7.
       ** For 'recursive' chunking, the `min_characters_per_chunk` is set to 24.
       ** For 'sdpm' chunking, the `similarity_threshold` is set to 0.7 and `skip_window` is set to 1.
       ** For 'neural' chunking, the `min_characters_per_chunk` is set to 24 and uses 'mirth/chonky_modernbert_base_1' model.
    """