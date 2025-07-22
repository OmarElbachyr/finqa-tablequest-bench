#!/usr/bin/env python3
"""
Script to generate token statistics for QA pairs in FinanceBench and TableQuest datasets.
"""

import json
import tiktoken
from collections import defaultdict

def count_tokens(text):
    """Count tokens in text using GPT-4 tokenizer."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def analyze_financebench():
    """Analyze FinanceBench dataset and return statistics by question_type."""
    print("Analyzing FinanceBench dataset...")
    
    stats = defaultdict(lambda: {'question_lengths': [], 'answer_lengths': [], 'count': 0})
    
    with open('../../datasets/financebench_open_source.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            question_type = data.get('question_type', 'unknown')
            question = data.get('question', '')
            answer = data.get('answer', '')
            
            q_tokens = count_tokens(question)
            a_tokens = count_tokens(answer)
            
            stats[question_type]['question_lengths'].append(q_tokens)
            stats[question_type]['answer_lengths'].append(a_tokens)
            stats[question_type]['count'] += 1
    
    # Calculate averages
    results = {}
    for q_type, data in stats.items():
        avg_q_len = sum(data['question_lengths']) / len(data['question_lengths']) if data['question_lengths'] else 0
        avg_a_len = sum(data['answer_lengths']) / len(data['answer_lengths']) if data['answer_lengths'] else 0
        results[q_type] = {
            'avg_question_length': avg_q_len,
            'avg_answer_length': avg_a_len,
            'count': data['count']
        }
    
    return results

def analyze_tablequest():
    """Analyze TableQuest dataset and return statistics by difficulty."""
    print("Analyzing TableQuest dataset...")
    
    stats = defaultdict(lambda: {'question_lengths': [], 'answer_lengths': [], 'count': 0})
    
    with open('data/csv/tq_document_qa_pairs.json', 'r') as f:
        data = json.load(f)
    
    for doc_data in data:
        for qa_pair in doc_data.get('qa_pairs', []):
            difficulty = qa_pair.get('difficulty', 'unknown')
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')
            
            q_tokens = count_tokens(question)
            a_tokens = count_tokens(answer)
            
            stats[difficulty]['question_lengths'].append(q_tokens)
            stats[difficulty]['answer_lengths'].append(a_tokens)
            stats[difficulty]['count'] += 1
    
    # Calculate averages
    results = {}
    for difficulty, data in stats.items():
        avg_q_len = sum(data['question_lengths']) / len(data['question_lengths']) if data['question_lengths'] else 0
        avg_a_len = sum(data['answer_lengths']) / len(data['answer_lengths']) if data['answer_lengths'] else 0
        results[difficulty] = {
            'avg_question_length': avg_q_len,
            'avg_answer_length': avg_a_len,
            'count': data['count']
        }
    
    return results

def format_table(title, stats, category_name):
    """Format statistics as a table."""
    table = f"\n{title}\n"
    table += "=" * len(title) + "\n"
    table += f"{'Category':<20} {'Avg. Question Length':<25} {'Avg. Answer Length':<20} {'QA Pair Count':<15}\n"
    table += "-" * 80 + "\n"
    
    total_count = 0
    for category, data in sorted(stats.items()):
        table += f"{category:<20} {data['avg_question_length']:<25.2f} {data['avg_answer_length']:<20.2f} {data['count']:<15}\n"
        total_count += data['count']
    
    table += "-" * 80 + "\n"
    table += f"{'Total':<20} {'--':<25} {'--':<20} {total_count:<15}\n"
    table += "\n"
    
    return table

def main():
    """Main function to generate and save statistics."""
    print("Generating QA Answer Token Statistics...")
    
    # Analyze datasets
    financebench_stats = analyze_financebench()
    tablequest_stats = analyze_tablequest()
    
    # Format tables
    fb_table = format_table("FinanceBench Statistics by Question Type", financebench_stats, "Question Type")
    tq_table = format_table("TableQuest Statistics by Difficulty", tablequest_stats, "Difficulty")
    
    # Print tables
    print(fb_table)
    print(tq_table)
    
    # Save to file
    output_file = "../../artifacts/datasets/qa_token_statistics.txt"
    with open(output_file, 'w') as f:
        f.write("QA Answer Token Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write("Generated statistics for token counts in QA pairs.\n")
        f.write("Token counts calculated using GPT-4 tokenizer.\n\n")
        f.write(fb_table)
        f.write(tq_table)
        
        # Additional detailed breakdown for TableQuest (matching the example format)
        if 'easy' in tablequest_stats and 'medium' in tablequest_stats and 'hard' in tablequest_stats:
            f.write("TableQuest Detailed Statistics (LaTeX format reference)\n")
            f.write("-" * 50 + "\n")
            f.write("Difficulty    Avg. Question Length    Avg. Answer Length    QA Pair Count\n")
            f.write("Easy          {:.2f}                  {:.2f}                {}\n".format(
                tablequest_stats['easy']['avg_question_length'],
                tablequest_stats['easy']['avg_answer_length'],
                tablequest_stats['easy']['count']
            ))
            f.write("Medium        {:.2f}                  {:.2f}                {}\n".format(
                tablequest_stats['medium']['avg_question_length'],
                tablequest_stats['medium']['avg_answer_length'],
                tablequest_stats['medium']['count']
            ))
            f.write("Hard          {:.2f}                  {:.2f}                {}\n".format(
                tablequest_stats['hard']['avg_question_length'],
                tablequest_stats['hard']['avg_answer_length'],
                tablequest_stats['hard']['count']
            ))
            total_tq = sum(data['count'] for data in tablequest_stats.values())
            f.write("Total         --                      --                    {}\n".format(total_tq))
    
    print(f"Statistics saved to: {output_file}")

if __name__ == "__main__":
    main()
