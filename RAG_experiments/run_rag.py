#!/usr/bin/env python
import os
import argparse
import pickle
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load API keys from environment variables
def load_environment():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables.")
    return openai_api_key, anthropic_api_key

from functions.combine_retriever import CombinedRetriever
from functions.ragbot_class import RagBot
from functions.rag_generation import get_rag_answer

# Load one or more BM25/Vectorstore retrievers
def load_retrievers():
    bm25_retriever_text = pickle.load(open("/workspace/RAG_experiments/data/Retriever/retriever_bm25_passage_small5k.pkl", "rb"))
    bm25_retriever_text.k = 10 # Set the number of top results to retrieve.

    return bm25_retriever_text # Return the retriever(s) you loaded (add more if needed)

def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG queries with a specified retriever.")
    parser.add_argument("--retriever", type=str, default="bm25_text",
                        choices=["vectorstore_text", "vectorstore_table", "vectorstore_combined", # Add/remove retrievers if needed
                                 "bm25_text", "bm25_table", "bm25_combined"],
                        help="Select which retriever to use.")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "ollama"],
                        help="Select the LLM provider (openai, anthropic or ollama for local models).")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", 
                        help="Select the language model.")
    parser.add_argument("--num_answers", type=int, default=1, 
                        help="Define the number of answers to generate.")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Indicate the temperature of the LLM")
    parser.add_argument("--text_ratio", type=float, default=0.5, # Only relevant if you use a combined retriever
                        help="Fraction of results that should be text (between 0 and 1).")
    parser.add_argument("--output_subfolder", type=str, default="MY_FIRST_RESULTS",
                        help="Name of the output subfolder for saving results.")
    parser.add_argument("--start_topic", type=int, default=1,
                        help="Start topic number.")
    parser.add_argument("--end_topic", type=int, default=51, # Adapt this value to the maximum number of queries you have
                        help="End topic number (exclusive).")
    return parser.parse_args()

def main():

    args = parse_args()

    # Load environment variables and API keys.
    openai_api_key, anthropic_api_key = load_environment()

    api_kwargs = {}
    if args.provider == "openai":
        api_kwargs["openai_api_key"] = openai_api_key
    elif args.provider == "anthropic":
        api_kwargs["anthropic_api_key"] = anthropic_api_key
    # For 'ollama', you might need to add a base URL.

    # Load topics (or any query file you use)
    with open('/workspace/RAG_experiments/data/Topics/topics-rnd5.xml', 'r') as f:
        topics = f.read()

    # Load the retrievers.
    bm25_retriever_text = load_retrievers() # Add more retrievers directly within the function if needed

    # # You can also create combined retrievers (e.g., for combined text & table results).
    # combined_bm25_retriever = CombinedRetriever(bm25_retriever_text, bm25_retriever_table, total_k=10, text_ratio=args.text_ratio)

    # Select the desired retriever.
    # Note: In this example, we only use the BM25 text retriever. Adapt this according to your needs.
    if args.retriever == "vectorstore_text":
        selected_retriever = vectorstore_retriever_text
    elif args.retriever == "vectorstore_table":
        selected_retriever = vectorstore_retriever_table
    elif args.retriever == "vectorstore_combined":
        selected_retriever = combined_vectorstore_retriever
    elif args.retriever == "bm25_text":
        selected_retriever = bm25_retriever_text
    elif args.retriever == "bm25_table":
        selected_retriever = bm25_retriever_table
    elif args.retriever == "bm25_combined":
        selected_retriever = combined_bm25_retriever
    else:
        raise ValueError("Invalid retriever selection.")

    # Initialize the RAG bot.
    rag_bot = RagBot(selected_retriever, 
                     provider=args.provider, 
                     model=args.model, 
                     num_answers=args.num_answers, 
                     temperature=args.temperature,
                     **api_kwargs,)

    # Loop through topics and generate answers.
    all_topic_data = []
    for topic_number in tqdm(range(args.start_topic, args.end_topic), desc="Processing Topics"):
        res = get_rag_answer(topics, topic_number, rag_bot)
        res['Retriever'] = args.retriever
        all_topic_data.append(res)
        # # uncomment the next lines to save your results after each iteration
        # all_topics_data_df = pd.DataFrame(all_topic_data)
        # all_topics_data_df.to_json("/workspace/RAG_experiments/res/stepwise_save.json", orient="records", indent=4)


    # Save results to a JSON file.
    detailed_df = pd.DataFrame(all_topic_data)
    output_dir = os.path.join("/workspace/RAG_experiments/res/", f"{args.output_subfolder}_res")
    os.makedirs(output_dir, exist_ok=True)
    detailed_file_path = os.path.join(output_dir, f"details_{args.output_subfolder}.json")
    detailed_df.to_json(detailed_file_path, orient="records", indent=4)
    print(f"Results saved to {detailed_file_path}")
    
if __name__ == "__main__":
    main()
