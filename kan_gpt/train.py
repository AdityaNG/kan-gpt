from kan_gpt.mingpt.utils import set_seed
from kan_gpt.dataset import WebTextDataset
from kan_gpt.mingpt.model import GPT
set_seed(3407)

model_type = 'gpt2'

# print an example instance of the dataset
train_dataset = WebTextDataset('test', model_type)
test_dataset = WebTextDataset('test', model_type)
x, y = train_dataset[0]
print(x.shape, y.shape)

# create a GPT instance
model_config = GPT.get_default_config()
model_config.model_type = model_type
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

# create a Trainer object
from kan_gpt.mingpt.trainer import Trainer

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
train_config.max_iters = 2000
train_config.num_workers = 0
trainer = Trainer(train_config, model, train_dataset)

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()

print("="*20)
print("EVAL")
print("="*20)

model.eval()

print("="*20)
