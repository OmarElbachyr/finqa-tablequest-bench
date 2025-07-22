import sys
root = "../.."
if root not in sys.path:
    sys.path.insert(0, root)

from retrievers.sentence_transformer import SentenceTransformerRetriever
from evaluation.classes.query_qrel_builder import QueryQrelsBuilder
from evaluation.classes.document_provider import DocumentProvider


if __name__ == "__main__":
    csv_path = "new_scripts/csv/link_qa_chunks.csv"

    provider = DocumentProvider(csv_path)
    print(f'Stats: {provider.stats}')
    
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    retriever = SentenceTransformerRetriever(provider, model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    run = retriever.search(queries, agg='max')
    # qid = 'q1'
    # print(qid, list(run[qid].items())[:5])

    retriever.evaluate(run, qrels, verbose=True)