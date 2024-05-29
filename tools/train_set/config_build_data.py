
class ConfigBuildData:
    
    def _get_empty_cfg_build_data() -> dict:
        return ConfigBuildData._get_cfg_build_data(
            cfg_name=None, base_cfg_path=None, dataset_cfg_path=None,
            num_classes=None,
            checkpoint_path=None, pretrain_dataset=None,
            save_best=None, save_interval=None, val_interval=None,
            batch_size=None, crop_size=None, iterations=None,
            epochs=None, dataset_name=None
        )
    
    @staticmethod
    def _get_cfg_build_data(
        cfg_name: str, base_cfg_path: str, dataset_cfg_path: str,
        num_classes: int,
        pretrained: bool, checkpoint_path: str, pretrain_dataset: str,
        save_best: bool, save_interval: int,
        val_interval: int, batch_size: int, crop_size: int,
        iterations: int, epochs: int, dataset_name: str
    )-> dict:
        return  {
                    "cfg_name"          :       cfg_name,
                    "base_cfg_path"     :       base_cfg_path,
                    "dataset_cfg_path"  :       dataset_cfg_path,
                    "num_classes"       :       num_classes,
                    "pretrained"        :       pretrained,
                    "checkpoint_path"   :       checkpoint_path,
                    "pretrain_dataset"  :       pretrain_dataset,
                    "save_best"         :       save_best,
                    "save_interval"     :       save_interval,
                    "val_interval"      :       val_interval,
                    "batch_size"        :       batch_size,
                    "crop_size"         :       crop_size,
                    "iterations"        :       iterations, 
                    "epochs"            :       epochs,
                    "dataset"           :       dataset_name    
                }