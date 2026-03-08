import torch
import torch.nn as nn


class MatrixFactorizationRouter(nn.Module):
    def __init__(self, num_models, dq, dm=128):
        super().__init__()
        self.model_embeddings = nn.Embedding(num_models, dm)
        self.W1 = nn.Linear(dq, dm)
        self.w2 = nn.Linear(dm, 1, bias=False)


    def delta(self, model_id, query_embedding):
        """
        Computes δ(M,q): δ(M,q) = w2^T ( vm ⊙ (W1^T vq + b) )
        """
        vm = self.model_embeddings(model_id)
        projected_query = self.W1(query_embedding)
        interaction = vm * projected_query
        score = self.w2(interaction)

        return score.squeeze(-1)


    def win_probability(self, model_a, model_b, query_embedding):
        """
        Implements equation: P(wins|q) = σ( δ(M,q) − δ(M',q) )
        """
        delta_a = self.delta(model_a, query_embedding)
        delta_b = self.delta(model_b, query_embedding)

        return torch.sigmoid(delta_a - delta_b)