import torch
import torch.nn.functional as F
import os

data_path = "./ft_diffs"
def get_diffs(layer, friendly_name):
    m1 = friendly_name + "_contrastive"
    m2 = friendly_name
    return torch.load(os.path.join(data_path, f"{m2}-{m1}_diffs_layer_{layer}.pt"))

def get_steering_vector_og_style(layer, friendly_name):
    vec = torch.load(os.path.join(data_path, f"{friendly_name}-vec_layer_{layer}.pt")).to(torch.float32)
    vec = vec / vec.norm()
    return vec.to(torch.float16)

def get_steering_vector(layer, friendly_name):
    vec = get_diffs(layer, friendly_name).to(torch.float32).mean(dim=0)
    vec = vec / vec.norm()
    return vec.to(torch.float16)

def get_activations(friendly_name, min_layer=13, max_layer=30):
    vecs = dict([(layer, []) for layer in range(min_layer, max_layer+1)])
    end_vecs = dict([(layer, []) for layer in range(min_layer, max_layer+1)])
    for layer in range(min_layer, max_layer+1):
        vecs[layer] = torch.load(os.path.join(data_path, f"{friendly_name}_vecs_layer_{layer}.pt"))
        end_vecs[layer] = torch.load(os.path.join(data_path, f"{friendly_name}_end_vecs_layer_{layer}.pt"))
    preds = torch.load(os.path.join(data_path, f"{friendly_name}_preds.pt"))
    correct = torch.load(os.path.join(data_path, f"{friendly_name}_correct.pt"))

    return vecs, end_vecs, preds, correct

def get_pre_post_activations(friendly_name, min_layer=13, max_layer=30):
    pre_activations = dict([(layer, []) for layer in range(min_layer, max_layer+1)])
    post_activations = dict([(layer, []) for layer in range(min_layer, max_layer+1)])

    for layer in range(min_layer, max_layer+1):
        pre_activations[layer] = torch.load(os.path.join(data_path, f"{friendly_name}_pre_activations_layer_{layer}.pt"))
        post_activations[layer] = torch.load(os.path.join(data_path, f"{friendly_name}_post_activations_layer_{layer}.pt"))

    continuations = torch.load(os.path.join(data_path, f"{friendly_name}_continuations.pt"))
    solutions = torch.load(os.path.join(data_path, f"{friendly_name}_solutions.pt"))

    return pre_activations, post_activations, continuations, solutions

def get_preds(friendly_name):
    return torch.load(os.path.join(data_path, f"{friendly_name}_preds.pt"))

def high_low_vector( zephyr, friendly_name, layer, multiplier, question, max_length=200 ):
    # Generate outputs
    zephyr.set_only_add_to_first_token(False)
    vec = get_steering_vector(layer, friendly_name)
    pos_multiplier = multiplier
    neg_multiplier = multiplier * -1.0
    zephyr.set_save_internal_decodings(False)

    zephyr_input = question

    zephyr.reset_all()
    zephyr.set_add_activations(layer, pos_multiplier * vec.cuda())
    sa = (
        zephyr.generate_text(zephyr_input, max_length=max_length).split("<|assistant|>")[-1].strip()
    )

    zephyr.reset_all()
    zephyr.set_add_activations(layer, neg_multiplier * vec.cuda())
    na = (
        zephyr.generate_text(zephyr_input, max_length=max_length).split("<|assistant|>")[-1].strip()
    )

    zephyr.reset_all()
    da = (
        zephyr.generate_text(zephyr_input, max_length=max_length).split("<|assistant|>")[-1].strip()
    )

    # Print generated outputs
    print("baseline:\n", da)
    print(f"\\+ {friendly_name}:\n", sa)
    print(f"\\- {friendly_name}:\n", na)

def calculateCosineSimilarity(matrix, key):
    key = key.to(torch.float32)
    matrix = matrix.to(torch.float32)

    # Ensure the vectors are in the correct shape
    key = F.normalize(key, p=2, dim=0)
    print( key.shape )

    sims = []
    for row in matrix:
        assert row.shape == key.shape
        row = F.normalize(row, p=2, dim=0)

        cos_sim = F.cosine_similarity(row, key)
        sims.append(cos_sim)
    return sims
