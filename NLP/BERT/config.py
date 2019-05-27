class Bert_base_model_cfg:
    dim = 768
    dim_ff = 3072
    n_layers = 12
    p_drop_attn = 0.1
    n_heads = 12
    p_drop_hidden = 0.1
    max_len = 512
    n_segments = 2
    vocab_size = 30522

class Bert_pretrain_cfg:
    seed = 3431
    batch_size = 96
    lr = 1e-4
    n_epochs = 25
    warmup = 0.1
    save_steps = 10000
    total_steps = 1000000

