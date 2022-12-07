import torch
import torch.nn as nn
from transformers import RobertaModel

class PQCLR(nn.Module):
    # Encodes all positives and anchors
    def __init__(self, device):
        super(PQCLR, self).__init__()
        self.passage_encoder = RobertaModel.from_pretrained('roberta-base').to(device)
        self.question_encoder = RobertaModel.from_pretrained('roberta-base').to(device)
    
    def forward(self, positives, anchors):
        # out_pos = self.passage_encoder(positives).pooler_output
        # out_anch = self.question_encoder(anchors).pooler_output

        # Using average of the output instead of cls
        out_pos = self.passage_encoder(positives).last_hidden_state
        out_anch = self.passage_encoder(anchors).last_hidden_state

        out_pos = torch.mean(out_pos, dim=1)
        out_anch = torch.mean(out_anch, dim=1)

        return out_pos, out_anch


class PQNCLR(nn.Module):
    # Encodes positives, anchors, and negatives
    def __init__(self, device):
        super(PQNCLR, self).__init__()
        if torch.cuda.device_count()>1:
            self.passage_encoder = torch.nn.DataParallel(RobertaModel.from_pretrained('roberta-base')).to(device)
            self.question_encoder = torch.nn.DataParallel(RobertaModel.from_pretrained('roberta-base')).to(device)
        else:
            self.passage_encoder = RobertaModel.from_pretrained('roberta-base').to(device)
            self.question_encoder = RobertaModel.from_pretrained('roberta-base').to(device)
    
    def forward(self, positives, anchors, negatives):
        # out_pos = self.passage_encoder(positives).pooler_output
        # out_anch = self.question_encoder(anchors).pooler_output

        # Using average of the output instead of cls
        out_pos = self.passage_encoder(positives).last_hidden_state
        out_anch = self.question_encoder(anchors).last_hidden_state
        out_neg = []
        for i in range(negatives.shape[1]):
            out_neg_i = self.passage_encoder(negatives[:,i,:]).last_hidden_state
            out_neg_i = torch.mean(out_neg_i, dim=1)
            out_neg.append(out_neg_i)

        # if len(out_neg) > 25:
        #     out_neg = out_neg[:25]

        out_pos = torch.mean(out_pos, dim=1)
        out_anch = torch.mean(out_anch, dim=1)
        # out_neg = torch.mean(out_neg, dim=1)
        out_neg = torch.stack(out_neg)

        return out_pos, out_anch, out_neg


