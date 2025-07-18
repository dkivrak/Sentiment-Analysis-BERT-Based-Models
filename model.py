from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    MBartForConditionalGeneration,
)


class TransformerModel:
    def __init__(self, model_name: str, task_type="classification", num_labels=2):
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels

    def get_model(self):
        if self.task_type == "classification":
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )

        elif self.task_type == "generation":
            if "t5" in self.model_name:
                return T5ForConditionalGeneration.from_pretrained(self.model_name)
            elif "mbart" in self.model_name:
                return MBartForConditionalGeneration.from_pretrained(self.model_name)
            else:
                raise ValueError(f"Unsupported generation model: {self.model_name}")

        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
