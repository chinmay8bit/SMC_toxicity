import torch
from transformers import RobertaTokenizer
from src.toxicity_classifier.modeling_roberta import RobertaForSequenceClassification

class ToxicityScorer:
    def __init__(self, label_idx=1, device='cuda'):
        """
        Initialize the ToxicityScorer with a model and tokenizer.
        :param model: Pretrained toxicity classification model.
        :param tokenizer: Tokenizer for the model.
        :param label_idx: Index of the toxicity label in the model's output.
        """
        self.tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
        self.model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier').to(device) # type: ignore
        self.device = device
        self.label_idx = label_idx

    def score_text(self, text):
        token_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        output = self.model(token_ids)
        logits = output.logits.log_softmax(dim=-1)
        return logits[..., self.label_idx]
    
    def score_token_ids(self, token_ids: torch.Tensor):
        output = self.model(token_ids)
        logits = output.logits.log_softmax(dim=-1)
        return logits[..., self.label_idx]
