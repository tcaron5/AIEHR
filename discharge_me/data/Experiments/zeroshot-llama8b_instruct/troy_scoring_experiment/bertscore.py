import torch
import torch.nn as nn
from bert_score import BERTScorer


class BertScore(nn.Module):
    def __init__(self):
        super(BertScore, self).__init__()
        with torch.no_grad():
            self.bert_scorer = BERTScorer(
                model_type="distilbert-base-uncased",
                num_layers=4,
                batch_size=8,
                nthreads=8,
                all_layers=False,
                idf=False,
                lang="en",
                rescale_with_baseline=True,
                baseline_path=None,
            )

    def forward(self, refs, hyps):
        p, r, f = self.bert_scorer.score(
            cands=hyps,
            refs=refs,
            verbose=False,
            batch_size=8,
        )
        return f.tolist()