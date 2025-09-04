import torch

def neg_partial_log_likelihood(log_risk, time, event):
    order = torch.argsort(time, descending=True)
    log_risk = log_risk[order]
    event = event[order]
    log_cumsum = torch.logcumsumexp(log_risk, dim=0)
    ll = (log_risk - log_cumsum) * event
    return - ll.sum() / (event.sum().clamp_min(1))
