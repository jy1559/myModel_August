import dataset
import model_code.model as model
from time import time
from loss import compute_loss
from tqdm.auto import tqdm
import os, torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)

train_loader, val_loader, test_loader = dataset.get_dataloaders(dataset_folder = '/home/jy1559/Mar2025_Module/Datasets',
                                            dataset_name='Globo',
                                              train_batch_th=10000,
                                              use_bucket_batching=True,
                                              use_add_info=True,)
dataset_name = 'Globo'
if dataset_name == 'Globo':
    add_info_num_cat = [('cat', 11), ('cat', 4), ('cat', 4), ('cat', 20), ('cat', 7), ('cat', 28)]
elif dataset_name == 'LFM-BeyMS':
    add_info_num_cat = []
elif dataset_name == 'Retail_Rocket':
    add_info_num_cat = [('cat', 3), ('num', -1)]

import dataset as dataset
llm_embedding = dataset.get_llm_embedding(dataset_folder = '/home/jy1559/Mar2025_Module/Datasets',
                                            dataset_name='Globo',)
num_items = llm_embedding['embedding_tensor'].shape[0]

model_config = {
    'dataset_name': 'Globo',
    'dt_method': 'bucket',
    'num_buckets': 32,
    'bucket_size': 2,
    'use_add_info': False,
    'use_llm': False,
    'use_dt': False,
    'num_layers': 2,
    'hidden_size': 512,
    'num_heads': 8,
    'dropout': 0.1,
    'add_info_specs': add_info_num_cat,
    'device': 'cuda:3',
}
mod = model.SeqRecModel(num_items, model_config).to('cuda:3')

i = 0
for batch in tqdm(train_loader, total=len(train_loader)):
    start_time = time()
    batch = {k: v.to("cuda:3") if torch.is_tensor(v) else v for k, v in batch.items()}
    out = mod(batch)
    loss = compute_loss(batch, out, mod, {'dataset_name': 'Globo'})
    end_time = time()
    if i % 500 == 0: print(f"Batch processed in {end_time - start_time:.4f} seconds, loss = {loss.item():.4f}")
    i += 1
    