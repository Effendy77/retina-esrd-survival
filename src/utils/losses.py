
import torch
def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    time = time[order]; event = event[order]; risk = risk[order]
    max_risk = torch.max(risk)
    exp_risk = torch.exp(risk - max_risk)
    cum_sum = torch.cumsum(exp_risk, dim=0)
    log_cum_sum = torch.log(cum_sum) + max_risk
    ll = (risk - log_cum_sum) * event
    return -torch.sum(ll) / torch.sum(event.clamp(min=1.0))
