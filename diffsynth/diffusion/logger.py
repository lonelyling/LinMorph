import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"new-step-{self.num_steps}.safetensors", save_full_model=True)


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
        accelerator.wait_for_everyone()


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors", save_full_model=True)

    def _filter_and_strip_prefix(self, state_dict):
        """
        Keep only keys under `remove_prefix_in_ckpt` and strip that prefix.
        If no key matches the prefix, return the original state_dict unchanged.
        """
        if self.remove_prefix_in_ckpt is None:
            return state_dict
        prefix = self.remove_prefix_in_ckpt
        filtered = {name[len(prefix):]: param for name, param in state_dict.items() if name.startswith(prefix)}
        if len(filtered) == 0:
            return state_dict
        return filtered


    # def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
    #     accelerator.wait_for_everyone()
    #     if accelerator.is_main_process:
    #         state_dict = accelerator.get_state_dict(model)
    #         state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
    #         state_dict = self.state_dict_converter(state_dict)
    #         os.makedirs(self.output_path, exist_ok=True)
    #         path = os.path.join(self.output_path, file_name)
    #         accelerator.save(state_dict, path, safe_serialization=True)

    def save_model(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        file_name,
        save_full_model=True,
        synchronize=True,
        trainable_filter=None,
    ):
        """
        保存模型权重
        Args:
            save_full_model: True 保存所有权重，False 只保存可训练权重（原来的行为）
            synchronize: True 时保持全 rank barrier；False 时只让主进程写盘，不做同步。
        """
        if synchronize:
            accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if save_full_model:
                # 1. 获取状态字典（完整模型）
                state_dict = accelerator.get_state_dict(model)
                # IMPORTANT:
                # Use accelerator.get_state_dict(model) for distributed training;
                # unwrap_model(...).state_dict() may be incomplete under ZeRO/FSDP.
                # For DiT checkpoints, keep only the configured prefix (e.g. "pipe.dit.")
                # and strip it so inference loader sees canonical key names.
                state_dict = self._filter_and_strip_prefix(state_dict)
            else:
                # 1. 轻量中间 checkpoint：直接导出本地可训练参数，避免 full state gather 峰值。
                local_state_dict = accelerator.unwrap_model(model).state_dict()
                state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                    local_state_dict, remove_prefix=self.remove_prefix_in_ckpt
                )
                if trainable_filter == "lora":
                    state_dict = {
                        name: parameter
                        for name, parameter in state_dict.items()
                        if ".lora_A." in name or ".lora_B." in name
                    }
                    if not state_dict:
                        raise RuntimeError("Requested a LoRA-only checkpoint, but no trainable LoRA tensors were found")
                elif trainable_filter is not None:
                    raise ValueError(f"Unsupported trainable_filter: {trainable_filter}")
            
            # 2. 应用转换器（如果有）
            if self.state_dict_converter is not None:
                state_dict = self.state_dict_converter(state_dict)
            
            # 3. 保存
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

            print(f"模型已保存到: {path}", flush=True)
            if save_full_model:
                print(f"保存了全部 {len(state_dict)} 个参数", flush=True)
            else:
                print(f"保存了可训练参数 {len(state_dict)} 个", flush=True)
        # 非同步模式用于 checkpoint 降级路径：只让主进程写盘，避免额外 barrier。
        if synchronize:
            accelerator.wait_for_everyone()
