# this is the file in which we create the infrastructure for training the model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import time

class Trainer:
    def __init__(self, model: nn.Module, dataset, epochs: int, max_seq: int, batch_size: int = 32, learning_rate: float = 6e-4):
        self.epochs = epochs
        self.max_seq = max_seq # Context length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # 1. Setup Device
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        
        # 2. DataLoader
        # Note: I separated batch_size from max_seq. 
        # max_seq is the length of one sentence; batch_size is how many sentences we process at once.
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 3. Optimizer with "GPT-2 style" Parameter Groups
        # We separate parameters: those that decay (weights) vs those that don't (biases, layernorms)
        self.optimizer = self._configure_optimizer(weight_decay=0.1)

    def _configure_optimizer(self, weight_decay):
        """
        Separates parameters into two groups:
        1. Weights (Linear, Conv1D) -> Apply Weight Decay
        2. Biases, LayerNorms, Embeddings -> No Weight Decay
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Embedding)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        
        # Create the pytorch optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        # GPT-2 Paper uses betas=(0.9, 0.95) instead of the default (0.9, 0.999)
        optimizer = AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer

    def _get_lr(self, it, total_iters, min_lr=6e-5, warmup_iters=100):
        """
        Calculates learning rate with Linear Warmup + Cosine Decay
        """
        # 1. Linear Warmup
        if it < warmup_iters:
            return self.learning_rate * (it + 1) / (warmup_iters + 1)
        
        # 2. If we are past the end, return min_lr
        if it > total_iters:
            return min_lr
            
        # 3. Cosine Decay
        decay_ratio = (it - warmup_iters) / (total_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (self.learning_rate - min_lr)

    def train(self):
        self.model.train()
        total_steps = self.epochs * len(self.dataloader)
        print(f"Starting training for {self.epochs} epochs ({total_steps} steps)...")
        
        step_count = 0
        t0 = time.time()
        
        for epoch in range(self.epochs):
            for batch_idx, (x, y) in enumerate(self.dataloader):
                # Move data to GPU/MPS
                x, y = x.to(self.device), y.to(self.device)
                
                # Update Learning Rate (Cosine Schedule)
                lr = self._get_lr(step_count, total_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # 1. Forward Pass
                # Model should return (logits, loss)
                logits, loss = self.model(x, y)
                
                # 2. Backward Pass
                self.optimizer.zero_grad(set_to_none=True) # Slightly more efficient than zero_grad()
                loss.backward()
                
                # 3. Gradient Clipping (GPT-2 Standard: Clip at norm 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 4. Update
                self.optimizer.step()
                
                step_count += 1
                
                # Logging
                if step_count % 100 == 0:
                    dt = time.time() - t0
                    print(f"Epoch {epoch+1} | Step {step_count}/{total_steps} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {dt:.2f}s")
                    t0 = time.time()

        print("Training Complete.")

    def save_model(self, path="gpt_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")