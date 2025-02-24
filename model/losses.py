"""
compute different loss
"""

import torch
import numpy as np

from torch import nn
import torch.nn.functional as F

def cross_entropy_loss(logits, targets):
    """ code taken from URL repo """
    log_p_y = F.log_softmax(logits, dim=1)
    preds = log_p_y.argmax(1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')
    acc = torch.eq(preds, labels).float().mean()
    
    return loss, acc
    
# NCC
def prototype_loss(support_embeddings, support_labels,
                   query_embeddings, query_labels, distance='cos'):
    """ code taken from URL repo """
    n_way = len(query_labels.unique())

    prots = compute_prototypes(support_embeddings, support_labels, n_way).unsqueeze(0)
    embeds = query_embeddings.unsqueeze(1)

    if distance == 'l2':
        logits = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', embeds, prots)
    elif distance == 'corr':
        logits = F.normalize((embeds * prots).sum(-1), dim=-1, p=2) * 10

    return cross_entropy_loss(logits, query_labels)



def compute_prototypes(embeddings, labels, n_way):
    """ code taken from URL repo """
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
        else:
            prots[i] = embeddings[(labels == i).nonzero(as_tuple=False), :].mean(0)
    return prots



