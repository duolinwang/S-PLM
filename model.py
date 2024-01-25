import torch
import yaml
from torch import nn
from torch.nn import functional as F
from collections.abc import Sequence
from transformers import EsmModel, T5Tokenizer, T5Model
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import load_configs, get_dummy_logging
import esm_adapterH
import esm
import numpy as np

def verify_data_types(model, logging):
    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        logging.info(f"{k}, {v}, {v / total}")


def prepare_hf_esm_model(model_name, configs, logging):
    if configs.encoder.quantization_4_bit:
        logging.info('load quantized 4-bit weights')
        # QLoRa fine-tuning:
        quantization_config = BitsAndBytesConfig(
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = EsmModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16
        )

        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )

    else:
        model = EsmModel.from_pretrained(model_name)

    if configs.encoder.lora.enable:
        config = LoraConfig(
            r=configs.encoder.lora.r,
            lora_alpha=configs.encoder.lora.lora_alpha,
            target_modules=[
                "query",
                "key",
                "value",
                "dense"
            ],
            inference_mode=False,
            lora_dropout=configs.encoder.lora.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)

        if configs.encoder.quantization_4_bit:
            logging.info('make embedding parameters trainable because of 4 bit training')
            for param in model.embeddings.word_embeddings.parameters():
                param.requires_grad = True

        verify_data_types(model, logging)

    elif not configs.encoder.quantization_4_bit and not configs.encoder.lora.enable and configs.encoder.fine_tune.enable:
        # fine-tune the latest layer

        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.encoder.layer[-configs.encoder.fine_tune.last_layers_trainable:].parameters():
            param.requires_grad = True

    else:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

    for param in model.pooler.parameters():
        param.requires_grad = False

    if configs.encoder.tune_embedding:
        logging.info('make embedding parameters trainable')
        for param in model.embeddings.word_embeddings.parameters():
            param.requires_grad = True

    if configs.encoder.fix_embedding:
        for name, param in model.named_parameters():
            param.requires_grad = False

    return model


def prepare_esm_model(configs, logging):
    logging.info("use ESM model")
    model_name = configs.encoder.model_name.split('/')[-1]

    # Create the model dynamically using module attributes
    model_constructor = getattr(esm.pretrained, model_name, None)
    model, alphabet = model_constructor()
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

        # only freeze all the parameters once at the beginning. then open some layers later

    if configs.encoder.lora.enable:
        logging.info('enable LoRa on top of esm model')
        target_modules = [
            "k_proj", "v_proj", "q_proj",
            "fc1", "fc2"
        ]
        config = LoraConfig(
            r=configs.encoder.lora.r,
            lora_alpha=configs.encoder.lora.lora_alpha,
            target_modules=target_modules,
            inference_mode=False,
            lora_dropout=configs.encoder.lora.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)

        verify_data_types(model, logging)

    elif not configs.encoder.lora.enable and configs.encoder.fine_tune.enable:
        # fine-tune the latest layer
        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.layers[-configs.encoder.fine_tune.last_layers_trainable:].parameters():
            param.requires_grad = True

        # if you need fine-tune last layer, the emb_layer_norm_after for last representation should be updated
        if configs.encoder.fine_tune.last_layers_trainable != 0:
            for param in model.emb_layer_norm_after.parameters():
                param.requires_grad = True

    if configs.encoder.tune_embedding:
        logging.info('make esm embedding parameters trainable')
        for param in model.embed_tokens.parameters():
            param.requires_grad = True

    return model, alphabet


def prepare_adapter_h_model(configs, logging):
    logging.info("use adapterH ESM model")

    adapter_args = configs.encoder.adapter_h
    model_name = configs.encoder.model_name.split('/')[-1]

    # Create the model dynamically using module attributes
    model_constructor = getattr(esm_adapterH.pretrained, model_name, None)
    model, alphabet = model_constructor(adapter_args)
    num_layers = model.num_layers
    # checkpoint = torch.load(configs.encoder.adapter_h.pretrain_checkpoints, map_location=lambda storage, loc: storage)
    # logging.info('load the pretrained weights of S-PLM from {configs.encoder.adapter_h.pretrain_checkpoints}')
    # model.load_state_dict(checkpoint['state_dict1'], strict=False)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    if configs.encoder.adapter_h.enable:
      if not isinstance(configs.encoder.adapter_h.freeze_adapter_layers, list):
        configs.encoder.adapter_h.freeze_adapter_layers = [configs.encoder.adapter_h.freeze_adapter_layers]
    
    if configs.encoder.fine_tune.enable:
      if not isinstance(configs.encoder.fine_tune.freeze_adapter_layers, list):
        configs.encoder.fine_tune.freeze_adapter_layers = [configs.encoder.fine_tune.freeze_adapter_layers]
    
    if configs.encoder.lora.enable:
        logging.info('enable LoRa on top of adapterH model')
        target_modules = []
        if configs.encoder.lora.esm_num_end_lora > 0:
            start_layer_idx = np.max([num_layers - configs.encoder.lora.esm_num_end_lora, 0])
            for idx in range(start_layer_idx, num_layers):
                for layer_name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                   "self_attn.out_proj"]:
                    target_modules.append(f"layers.{idx}.{layer_name}")
        
        config = LoraConfig(
            r=configs.encoder.lora.r,
            lora_alpha=configs.encoder.lora.lora_alpha,
            target_modules=target_modules,
            inference_mode=False,
            lora_dropout=configs.encoder.lora.lora_dropout,
            bias="none",
            #modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, config)

        verify_data_types(model, logging)

    elif not configs.encoder.lora.enable and configs.encoder.fine_tune.enable:
        # fine-tune the latest layer

        # Allow the parameters of the last transformer block to be updated during fine-tuning
        for param in model.layers[-configs.encoder.fine_tune.last_layers_trainable:].parameters():
            param.requires_grad = True

        # if you need fine-tune last layer, the emb_layer_norm_after for last representation should be updated
        if configs.encoder.fine_tune.last_layers_trainable != 0:
            for param in model.emb_layer_norm_after.parameters():
                param.requires_grad = True
    
    
    # only freeze all the parameters once at the beginning. then open some layers later
    #only make adapterH trainable according to freeze_adapter_layers
    if configs.encoder.adapter_h.enable:
      for adapter_idx, value in enumerate(configs.encoder.adapter_h.freeze_adapter_layers):
        if not value:
            for name, param in model.named_parameters():
                adapter_name = f"adapter_{adapter_idx}"
                if adapter_name in name:
                    # Freeze all parameters by default
                    param.requires_grad = True
    
    # only freeze all the parameters once at the beginning. then open some layers later,but because
    # of fine_tune, adapter layers might be tunable.
    #change on 1/15/2024 not need to use freeze_adapter_layers to control fine-tune part! use another parameter instead and must after setting of freeze_adapter_layers
    if configs.encoder.fine_tune.enable: #only see fine_tune.freeze_adapter_layers when fine-tune is available
       for adapter_idx, value in enumerate(configs.encoder.fine_tune.freeze_adapter_layers):
        if value:
            for name, param in model.named_parameters():
                adapter_name = f"adapter_{adapter_idx}"
                if adapter_name in name:
                    # Freeze all parameters by default
                    print("freeze adapter in fine-tune")
                    param.requires_grad = False
    #"""
    
    if configs.encoder.tune_embedding:
        for param in model.embed_tokens.parameters():
            param.requires_grad = True

    return model, alphabet


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        hidden = F.softmax(hidden, dim=-1)
        return hidden


class Encoder(nn.Module):
    def __init__(self, logging, configs):
        super().__init__()
        if configs.encoder.adapter_h.enable:
            self.esm2, self.alphabet = prepare_adapter_h_model(configs, logging)
        else:
            self.esm2, self.alphabet = prepare_esm_model(configs, logging)

        self.head = nn.Linear(self.esm2.embed_dim, configs.encoder.num_classes)
        self.pooling_layer = nn.AdaptiveAvgPool1d(output_size=1)
        # self.device = device
        self.configs = configs

    def forward(self, x):
        features = self.esm2(x['input_ids'],
                             repr_layers=[self.esm2.num_layers])['representations'][self.esm2.num_layers]

        transposed_feature = features.transpose(1, 2)
        pooled_features = self.pooling_layer(transposed_feature).squeeze(2)
        classification = self.head(pooled_features)
        return classification


class EncoderSSPTM(nn.Module):
    def __init__(self, logging, configs):
        super().__init__()
        if configs.encoder.adapter_h.enable:
            self.esm2, self.alphabet = prepare_adapter_h_model(configs, logging)
        else:
            self.esm2, self.alphabet = prepare_esm_model(configs, logging)

        # extract the embedding size
        mlp_input_dim = self.esm2.embed_dim

        mlp_hidden_dim = configs.encoder.mlp_hidden_dim
        mlp_layer_num = configs.encoder.mlp_layer_num
        hidden_dims = [mlp_hidden_dim] * (mlp_layer_num - 1)
        self.mlp = MultiLayerPerceptron(mlp_input_dim, hidden_dims + [configs.encoder.num_classes], batch_norm=False,
                                        dropout=configs.encoder.head_dropout)

        # self.device = device
        self.configs = configs

    def forward(self, x):
        features = self.esm2(x['input_ids'],
                            repr_layers=[self.esm2.num_layers])['representations'][self.esm2.num_layers]
        c = self.mlp(features[:, 1:-1, :])
        return c


def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model, logging):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)
    logging.info(
        f"trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_models(configs, logging):
    """
    Prepare the encoder model.

    Args:
        configs: A python box object containing the configuration options.
        logging: The logging object.

    Returns:
        The encoder model.
    """
    # Prepare the encoder.
    encoder = Encoder(logging=logging, configs=configs)
    print_trainable_parameters(encoder, logging)
    logging.info('encoder parameters: ' + str(sum(p.numel() for p in encoder.parameters())))

    return encoder


def prepare_models_secondary_structure_ptm(configs, logging):
    """
    Prepare the encoder model.

    Args:
        configs: A python box object containing the configuration options.
        logging: The logging object.

    Returns:
        The encoder model.
    """
    # Prepare the encoder.
    encoder = EncoderSSPTM(logging=logging, configs=configs)
    print_trainable_parameters(encoder, logging)
    logging.info('encoder parameters: ' + str(sum(p.numel() for p in encoder.parameters())))

    return encoder


if __name__ == '__main__':
    # For test model and its modules
    config_path = './config.yaml'
    with open(config_path) as file:
        configs_dict = yaml.full_load(file)

    configs_file = load_configs(configs_dict)

    dummy_logging = get_dummy_logging()

    encoder_model = prepare_models(configs_file, dummy_logging)
    input_tensor = torch.randint(high=30, low=0, size=(2, 1024), dtype=torch.int64)

    sample = {'input_ids': input_tensor, 'attention_mask': torch.ones(input_tensor.shape)}
    output = encoder_model(sample)
    print(output.shape)
    print('done')
