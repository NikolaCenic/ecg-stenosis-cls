import torch
import torch.nn.functional as F


class CLIPLoss(torch.nn.Module):
    def __init__(self, temperature, lambda_0=0.5):
        super().__init__()
        self.lambda_0 = lambda_0
        self.lambda_1 = 1 - lambda_0
        self.temperature = temperature

    def forward(self, ecg_emb, angio_emb):
        """
        img_emb:  (B, D) image embeddings
        text_emb: (B, D) text embeddings
        """
        # normalize
        ecg_emb = F.normalize(ecg_emb, dim=-1).float()
        angio_emb = F.normalize(angio_emb, dim=-1).float()

        # compute similarity
        logits = (ecg_emb @ angio_emb.T) / self.temperature
        targets = torch.arange(len(ecg_emb), device=ecg_emb.device)

        # cross entropy in both directions
        loss_0 = self.lambda_0 * F.cross_entropy(logits, targets)
        loss_1 = self.lambda_1 * F.cross_entropy(logits.T, targets)

        loss = loss_0 + loss_1
        return loss, logits
