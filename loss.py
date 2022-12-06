
import torch
import torch.nn.functional as F

# Taken from https://github.com/edreisMD/ConVIRT-pytorch/blob/a614f2e5b381d2d4c3ef6d80f3170796b0fc964f/loss/nt_xent.py#L4

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, alpha_weight):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs,
                    norm=True,
                    weights=1.0):
        temperature = self.temperature
        alpha = self.alpha_weight

        """
        Pytorch implementation of the loss  SimCRL function by googleresearch: https://github.com/google-research/simclr
        @article{chen2020simple,
                title={A Simple Framework for Contrastive Learning of Visual Representations},
                author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2002.05709},
                year={2020}
                }
        @article{chen2020big,
                title={Big Self-Supervised Models are Strong Semi-Supervised Learners},
                author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2006.10029},
                year={2020}
                }
        """

        LARGE_NUM = 1e9
        """Compute loss for model.
        Args:
        hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        tpu_context: context information for tpu.
        weights: a weighting number or vector.
        Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
        """
        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
            
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = labels.to(self.device)
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        
        """
        Different from Image-Image contrastive learning
        In the case of Image-Text contrastive learning we do not compute the similarity function between the Image-Image and Text-Text pairs  
        """
        # logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,0, 1)) / temperature
        # logits_aa = logits_aa - masks * LARGE_NUM
        # logits_bb = torch.matmul(hidden2,  torch.transpose(hidden2_large,0, 1)) / temperature
        # logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large,0, 1)) / temperature

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)

        return alpha*loss_a + (1-alpha)*loss_b



class CustomNegLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, alpha_weight):
        super(CustomNegLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, ps, qs, ns,
                    norm=True,
                    weights=1.0):
        temperature = self.temperature
        alpha = self.alpha_weight

        """
        Pytorch implementation of the loss  SimCRL function by googleresearch: https://github.com/google-research/simclr
        @article{chen2020simple,
                title={A Simple Framework for Contrastive Learning of Visual Representations},
                author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2002.05709},
                year={2020}
                }
        @article{chen2020big,
                title={Big Self-Supervised Models are Strong Semi-Supervised Learners},
                author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2006.10029},
                year={2020}
                }
        """

        LARGE_NUM = 1e9
        """Compute loss for model.
        Args:
        hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        tpu_context: context information for tpu.
        weights: a weighting number or vector.
        Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
        """
        # Get (normalized) hidden1 and hidden2.
        if norm:
            ps = F.normalize(ps, p=2, dim=1)
            qs = F.normalize(qs, p=2, dim=1)
            ns = F.normalize(ns, p=2, dim=2)
            # zis = F.normalize(zis, p=2, dim=1)
            # zjs = F.normalize(zjs, p=2, dim=1)
            
        # hidden1, hidden2 = zis, zjs
        # batch_size = hidden1.shape[0]
        batch_size = ps.shape[0]
        len_negatives = ns.shape[0]
        len_emb = ns.shape[2]

        qs = qs.unsqueeze(1)
        qs = torch.transpose(qs, 1,2)
        ps = ps.unsqueeze(1)

        ns = ns.reshape((batch_size, len_negatives, len_emb))
        
        pos_prod = torch.bmm(ps, qs)
        neg_prod = torch.bmm(ns, qs)
        
        pos_prod = pos_prod.squeeze(2)
        neg_prod = neg_prod.squeeze(2)

        logits = torch.concat((pos_prod,neg_prod), dim=1) / temperature
        loss_val = -1*torch.nn.functional.log_softmax(logits, dim=1)
        out_tensor = torch.zeros((len_negatives+1,1)).to(self.device)
        out_tensor[0,0] += 1
        out_val = torch.matmul(loss_val, out_tensor).sum()
        return out_val

        # logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,0, 1)) / temperature
        # logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large,0, 1)) / temperature

        # loss_a = self.softXEnt(labels, logits_ab)
        # loss_b = self.softXEnt(labels, logits_ba)



        # return alpha*loss_a + (1-alpha)*loss_b
