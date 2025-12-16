from dataset import *
from train import *
from gpt_model import *



# dataset
dataset = ShakespeareDataset(block_size=512)

print(dataset.data)
# model
main_model = GPT(
    vocab_size=66,
    seq_len=512,
    n_layers=6,
    d_model=348,
    d_inner=3*348,
    n_heads=6,
    dropout=0.2
)
n_params = sum(p.numel() for p in main_model.parameters())
print("number of parameters: %.2fM" % (n_params/1e6,))

# trainer
trainer = Trainer(
    model=main_model,
    dataset=dataset,
    epochs=1,
    max_seq=512
)

# # train
trainer.train()