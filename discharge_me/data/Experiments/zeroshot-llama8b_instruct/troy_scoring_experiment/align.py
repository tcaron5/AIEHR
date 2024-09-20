import torch
import torch.nn as nn
from alignscore import AlignScore

class AlignScorer(nn.Module):
    def __init__(self):
        super(AlignScorer, self).__init__()
        self.align_scorer = AlignScore(
            model='roberta-base', 
            device='cpu',
            batch_size=8, 
            ckpt_path='https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt', 
            evaluation_mode='nli_sp')

    def forward(self, refs, hyps):
        f = self.align_scorer.score(
            contexts=refs,
            claims=hyps,
        )
        return f
