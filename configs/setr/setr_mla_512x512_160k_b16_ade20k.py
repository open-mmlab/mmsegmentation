_base_ = ['./setr_mla_512x512_160k_ade20k_bs_8.py']

# num_gpus: 8 -> batch_size: 16
data = dict(samples_per_gpu=2)
