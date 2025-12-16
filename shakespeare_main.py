from dataset import *
from train import *
from gpt_model import *



# dataset
dataset = ShakespeareDataset(block_size=512)

print(dataset.data)
# model
main_model = GPT(
    vocab_size=1024,
    seq_len=512,
    n_layers=3,
    d_model=48,
    d_inner=3*48,
    n_heads=3,
    dropout=0.1
)
n_params = sum(p.numel() for p in main_model.parameters())
print("number of parameters: %.2fM" % (n_params/1e6,))

# trainer
trainer = Trainer(
    dataset=dataset,
    epochs=1,
    max_seq=512
)

# train
trainer.train()