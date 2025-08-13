import json
from pathlib import Path
from typing import Dict, List

class EvaluationDataset:

    """
    Represents a dataset for evaluating a retriever's performance.
    The dataset should be in JSON format with the following structure:
    The srt file must be in the same folder as the dataset file
    {
        "config": {
            "srt_file": "algebra1_overlap.srt",
            "lecture_name" : "algebra1"
        },
        "questions": [
            {
                "question": "What is a prime?",
                "relevant_documents": ["13": 0.9, "15": 0.4]
            },
            ...
        ]
    }
    """

    def __init__(self, dataset_file: str):
        file = Path(dataset_file)

        if not file.exists():
            raise FileNotFoundError(f"Dataset file '{file}' does not exist.")

        with open(file, 'r') as dataset:
            data = json.load(dataset)
        
        self.validate_dataset(data)

        self.data = data
        self.file = file
    
    @staticmethod
    def validate_dataset(dataset):
        required_keys = ["config", "questions"]
        for key in required_keys:
            if key not in dataset:
                raise ValueError(f"Dataset is missing required key: {key}")
        
        required_config_keys = ["srt_file", "lecture_name"]
        for key in required_config_keys:
            if key not in dataset["config"]:
                raise ValueError(f"Dataset config is missing required key: {key}")
        
        required_question_keys = ["question", "relevant_documents"]
        for i, question in enumerate(dataset["questions"]):
            for key in required_question_keys:
                if key not in question:
                    raise ValueError(f"Question {i} is missing required key: {key}")
    
    @property
    def absolute_srt_path(self) -> Path:
        """
        Returns the absolute path to the SRT file specified in the dataset config.
        """
        srt_file = self.data["config"]["srt_file"]
        return self.file.parent / srt_file

    @property
    def questions(self) -> List[Dict]:
        return self.data["questions"]

    def get_question_text(self, question_idx: int) -> str:
        return self.questions[question_idx]["question"]
    
    def get_relevant_documents(self, question_idx: int) -> Dict[str, float]:
        relevant_docs = self.questions[question_idx]["relevant_documents"]
        return {int(doc_id) : score for doc_id, score in relevant_docs.items()}
    
    def __len__(self) -> int:
        return len(self.questions)