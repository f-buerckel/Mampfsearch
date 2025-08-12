import click
import csv
from datetime import datetime
from mampfsearch.benchmark import Benchmark, EvaluationDataset
from mampfsearch.retrievers import DenseRetriever, HybridRetriever, HybridColbertRerankingRetriever, RerankerRetriever
from mampfsearch.utils import config
from pathlib import Path

@click.command("benchmark")
def benchmark():
    run()


def run():

    from rerankers import Reranker

    dataset_paths = [
        #config.get_benchmark_path() / "algebra1" / "algebra1.json",
        #config.get_benchmark_path() / "mle" / "backpropagation.json",
        config.get_benchmark_path() / "mle" / "backpropagation-keywords.json",
        #config.get_benchmark_path() / "mle" / "convolutional_neural_networks.json",
        #config.get_benchmark_path() / "mle" / "cross_validation.json",
        # config.get_benchmark_path() / "mle" / "data_augmentation_self_supervised_learning.json",
        # config.get_benchmark_path() / "mle" / "fully_connected_nets.json"
    ]

    retrievers = [
        DenseRetriever(),
        HybridRetriever(),
        #HybridColbertRerankingRetriever(),
    ]

    rerankers = {
        "bge-reranker": Reranker('BAAI/bge-reranker-v2-m3', verbose=False),
        #"mxbai-rerank-large": Reranker('mixedbread-ai/mxbai-rerank-large-v1', model_type='cross-encoder', verbose=False),
        "ms-marco-MiniLM-L12-v2": Reranker('cross-encoder/ms-marco-MiniLM-L12-v2', model_type='cross-encoder', verbose=False),
    }

    results = []


    for dataset_path in dataset_paths:
        for retriever in retrievers:
            dataset_name = dataset_path.stem
            retriever_name = retriever.__class__.__name__
            click.echo(f"Running benchmark for dataset: {dataset_name}, retriever: {retriever_name}")
            dataset = EvaluationDataset(dataset_path)
            benchmark = Benchmark(eval_dataset=dataset, retriever=retriever, name=dataset_name)
            result = benchmark.run()

            results.append({
                'dataset':dataset_name,
                'retriever': retriever_name,
                'reranker': 'None',
                'average_score': result['average_score'],
                'duration_seconds': result['duration_seconds'],
                'time_per_question': result['time_per_question']
            })

            for reranker_name, reranker in rerankers.items():
                click.echo(f"Running benchmark for dataset: {dataset_name}, retriever: {retriever_name}, reranker: {reranker_name}")
                reranked_benchmark = Benchmark(eval_dataset=dataset, retriever=RerankerRetriever(base_retriever=retriever, reranker=reranker), name=dataset_name)
                result = reranked_benchmark.run()

                results.append({
                    'dataset': dataset_name,
                    'retriever': retriever_name,
                    'reranker': reranker_name,
                    'average_score': result['average_score'],
                    'duration_seconds': result['duration_seconds'],
                    'time_per_question': result['time_per_question']
                })
    
    save_results_to_csv(results)

def save_results_to_csv(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_results_{timestamp}.csv"
    csv_path = Path(csv_filename)
    
    fieldnames = ['dataset', 'retriever', 'reranker', 'average_score', 'duration_seconds', 'time_per_question']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
