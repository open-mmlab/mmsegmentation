import os
import mmengine
from mmdet3d.registry import HOOKS
from mmengine.hooks.hook import Hook
from datetime import datetime
from mmdet3d.objnn_utils import TMP_OBJNN_PATH, TMP_ANCHOR_PATH, transfer_objnn_sigmoid


@HOOKS.register_module()
class GeneratePseudoAnomalyHook(Hook):
    priority = 'VERY_LOW'
    
    # def before_run(self, runner):
    #     if os.path.exists(TMP_OBJNN_PATH):
    #         os.remove(TMP_OBJNN_PATH)
    #     if os.path.exists(TMP_ANCHOR_PATH):
    #         os.remove(TMP_ANCHOR_PATH)
    #     grid_config = runner.cfg['grid_config']
    #     anchor = runner.model.initialize_objnn(runner.train_dataloader.dataset, grid_config)
    #     mmengine.dump(anchor, TMP_ANCHOR_PATH)
    
    def before_train_epoch(self, runner):
        prompt = 'a photo of a dog'
        a_promt = 'best quality'
        n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
        num_samples = 4
        image_resolution = 256
        detect_resolution = 512
        ddim_steps = 20
        control_start_step = 10
        control_end_step = 20
        guess_mode = False
        self_control = True
        strength = 1.0
        scale = 9.0
        seed = int(time.time()) % 100000
        eta = 1.0
        runner.model.ldm.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        img, intermediates = runner.model.ldm.sample_create_image_mask(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond, 
                                                        control_start_step=control_start_step, 
                                                        control_end_step=control_end_step, 
                                                        self_control=self_control)
        from PIL import Image
        for i, sample in enumerate(img):
            Image.fromarray(sample.cpu().numpy()).save(f'{seed}_{i}.jpg')
    
    def after_train_iter(self, 
                         runner,
                         batch_idx, 
                         data_batch, 
                         outputs):
        
        path = os.path.join(runner.work_dir, f'refine_{runner.timestamp}')
        runner.visualization_fov(runner.model, data_batch, path)
        runner.visualization_bev(runner.model.grid_config, data_batch, path)
        time_objnns = runner.model.objnns
        objnn_update = [[None for _ in objnns] for objnns in time_objnns]
        for ti, objnns in enumerate(time_objnns):
            for oi, objnn in enumerate(objnns):
                if (objnn.delta_obj_trans-0).abs().sum() > 1e-5 or (objnn.delta_obj_yaw-0).abs().sum() > 1e-5:
                    obj_trans, obj_yaw = transfer_objnn_sigmoid(objnn)
                    objnn_update[ti][oi] = (obj_trans.squeeze(-1).detach().cpu().numpy(), obj_yaw.squeeze(-1).detach().cpu().numpy())
        mmengine.dump(objnn_update, TMP_OBJNN_PATH)