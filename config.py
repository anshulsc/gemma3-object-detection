from dataclasses import dataclass

import torch


@dataclass
class Configuration:
    dataset_id: str = "ariG23498/license-detection-paligemma"

    model_id: str = "google/gemma-3-4b-pt"
    checkpoint_id: str = "sergiopaniego/gemma-3-4b-pt-object-detection-aug"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    batch_size: int = 8
    learning_rate: float = 2e-05
    epochs = 2
    
    # training type : "qlora", "lora", "full"
    training_type: str = "full"
    
    lora_r: int = 16 
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    project_name: str = "gemma3-object-detection"
    run_name: str = "full"
    
    
