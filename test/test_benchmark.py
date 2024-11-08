import unittest
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask

from deepeval.benchmarks import MMLU
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.models import DeepEvalBaseLLM

from main import RAG, run_query_sync


class RAGWrapper(DeepEvalBaseLLM):
    def __init__(self):
        self.rag = RAG()
        self.rag.on_startup()

    def generate(self, prompt: str, **kwargs) -> str:
        response = run_query_sync(prompt)
        return response

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def get_model_name(self) -> str:
        return "RAGWrapper"

    def load_model(self):
        pass

    def shutdown(self):
        self.rag.on_shutdown()

def run_mmlu_benchmark():
    # Define the benchmark
    benchmark = MMLU(
        tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
        n_shots=3
    )

    # Initialize RAG wrapped model
    rag_model = RAGWrapper()

    # Evaluate the model using MMLU benchmark
    benchmark.evaluate(model=rag_model)

    # Output results
    print(f"Overall Score: {benchmark.overall_score}")

    # Shutdown the model after use
    rag_model.shutdown()


class TestMMLUBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.benchmark = MMLU(
            tasks=[MMLUTask.MANAGEMENT, MMLUTask.MISCELLANEOUS, MMLUTask.MARKETING],
            n_shots=3
        )
        cls.rag_model = RAGWrapper()

    @classmethod
    def tearDownClass(cls):
        cls.rag_model.shutdown()

    def test_mmlu_benchmark(self):
        self.benchmark.evaluate(model=self.rag_model)
        overall_score = self.benchmark.overall_score
        self.assertGreaterEqual(overall_score, 0, "Overall score should be at least 0.")
        self.assertLessEqual(overall_score, 1, "Overall score should not exceed 1.")
        print(f"Overall Score: {overall_score}")

if __name__ == '__main__':
    unittest.main()