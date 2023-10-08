# model settings
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0)

# adapted from stable-diffusion/configs/stable-diffusion/v1-inference.yaml
stable_diffusion_cfg = dict(
    base_learning_rate=0.0001,
    target='ldm.models.diffusion.ddpm.LatentDiffusion',
    checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/'
    'vpd/stable_diffusion_v1-5_pretrain_third_party.pth',
    params=dict(
        linear_start=0.00085,
        linear_end=0.012,
        num_timesteps_cond=1,
        log_every_t=200,
        timesteps=1000,
        first_stage_key='jpg',
        cond_stage_key='txt',
        image_size=64,
        channels=4,
        cond_stage_trainable=False,
        conditioning_key='crossattn',
        monitor='val/loss_simple_ema',
        scale_factor=0.18215,
        use_ema=False,
        scheduler_config=dict(
            target='ldm.lr_scheduler.LambdaLinearScheduler',
            params=dict(
                warm_up_steps=[10000],
                cycle_lengths=[10000000000000],
                f_start=[1e-06],
                f_max=[1.0],
                f_min=[1.0])),
        unet_config=dict(
            target='ldm.modules.diffusionmodules.openaimodel.UNetModel',
            params=dict(
                image_size=32,
                in_channels=4,
                out_channels=4,
                model_channels=320,
                attention_resolutions=[4, 2, 1],
                num_res_blocks=2,
                channel_mult=[1, 2, 4, 4],
                num_heads=8,
                use_spatial_transformer=True,
                transformer_depth=1,
                context_dim=768,
                use_checkpoint=True,
                legacy=False)),
        first_stage_config=dict(
            target='ldm.models.autoencoder.AutoencoderKL',
            params=dict(
                embed_dim=4,
                monitor='val/rec_loss',
                ddconfig=dict(
                    double_z=True,
                    z_channels=4,
                    resolution=256,
                    in_channels=3,
                    out_ch=3,
                    ch=128,
                    ch_mult=[1, 2, 4, 4],
                    num_res_blocks=2,
                    attn_resolutions=[],
                    dropout=0.0),
                lossconfig=dict(target='torch.nn.Identity'))),
        cond_stage_config=dict(
            target='ldm.modules.encoders.modules.AbstractEncoder')))

model = dict(
    type='DepthEstimator',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='VPD',
        diffusion_cfg=stable_diffusion_cfg,
    ),
)

# some of the parameters in stable-diffusion model will not be updated
# during training
find_unused_parameters = True
