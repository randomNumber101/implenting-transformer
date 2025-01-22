from torch.optim import AdamW


class NoamLR:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        scale = self.d_model ** -0.5
        step_factor = min(self.current_step ** -0.5, (self.current_step + 1) * (self.warmup_steps ** -1.5))
        return scale * step_factor


def initialize_optimizer_and_scheduler(model, d_model, warmup_steps, lr=1e-3, weight_decay=0.01):
    # Filter parameters to exclude bias and layer norm from weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = NoamLR(optimizer, d_model, warmup_steps)

    return optimizer, scheduler
