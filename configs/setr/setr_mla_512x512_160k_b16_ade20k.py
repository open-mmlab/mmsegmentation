_base_ = ['./setr_mla_512x512_160k_b8_ade20k.py']

# num_gpus: 8 -> batch_size: 16
data = dict(samples_per_gpu=2)
