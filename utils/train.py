from utils.model import buildModel, buildTokenizer, WrappedZephyr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from math import log
from matplotlib.ticker import ScalarFormatter

data_path = "./ft_diffs"

assistant_tokens = torch.Tensor([28789, 28766, 489, 11143, 28766, 28767, 13]).to('cuda')

def findLastOccurrence(tensor_2d, sequence):
    seq_len = len(sequence)
    last_occurrences = []

    for row in tensor_2d:
        last_occurrence = -1
        for i in range(len(row) - seq_len, -1, -1):
            if torch.all(row[i:i + seq_len] == torch.tensor(sequence)):
                last_occurrence = i+seq_len-1
                break
        last_occurrences.append(last_occurrence)

    return last_occurrences

def loss_fn( logits, ground_truths ):
    """
    CE loss on all tokens in the assistant's response
    """
    last_indices = findLastOccurrence(ground_truths, assistant_tokens)

    loss = 0
    for i, row in enumerate(logits):
        assert last_indices[i] != -1
        row = row[last_indices[i]:-1, :]
        logprobs = torch.nn.functional.log_softmax( row, dim=-1 )

        ground_truth = ground_truths[i, last_indices[i]+1:]
        l = -1 * logprobs.gather(-1, ground_truth.unsqueeze(-1).to(torch.int64)).mean()
        loss += l

    return loss / len(logits)

class CustomTrainer(Trainer):
    """
    don't use labels because they're junk
    instead do cross-entroy on input_ids
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels
        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(input_ids=inputs["input_ids"])

        # Custom loss computation
        logits = outputs.get("logits")
        custom_loss = loss_fn(logits, inputs["input_ids"])

        # Optionally, return outputs for further use
        return (custom_loss, outputs) if return_outputs else custom_loss

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def train( dataset, epochs, batch_size, output_dir, batch_accumulation=1):
    tokenizer = buildTokenizer()

    model = buildModel(checkpoint="HuggingFaceH4/zephyr-7b-beta")
    pipe = pipeline("text-generation", model=model, torch_dtype=torch.bfloat16, device_map="auto", tokenizer=tokenizer)
    model = pipe.model
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    print_trainable_parameters(model)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    # Apply the accelerator. You can comment this out to remove the accelerator.
    model = accelerator.prepare_model(model)
    model = model.to('cuda')

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2.5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=batch_accumulation,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to=[]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

def fineTune(datasets, checkpoint_root, batch_size, epochs, batch_accumulation=1):
    dataset, contrastive_dataset = datasets
    train(dataset=dataset, batch_size=batch_size, batch_accumulation=batch_accumulation, epochs=epochs, output_dir=checkpoint_root)
    train(dataset=contrastive_dataset, batch_size=batch_size, batch_accumulation=batch_accumulation, epochs=epochs, output_dir=checkpoint_root+"_contrastive")

def extractDiffsOgStyle( dataset, checkpoint, friendly_name, start_layer=13, end_layer=30 ):
    model = WrappedZephyr(system_prompt="You are a helpful, honest, and concise assistant.", checkpoint=checkpoint)

    layers = list(range(start_layer, end_layer + 1))
    diffs = dict([(layer, []) for layer in layers])
    model.set_save_internal_decodings(False)
    model.reset_all()
    for datum in tqdm(dataset, desc="Processing prompts"):
        s_tokens = datum["s_tokens"]
        n_tokens = datum["n_tokens"]
        s_tokens = s_tokens.to(model.device).unsqueeze(0)
        n_tokens = n_tokens.to(model.device).unsqueeze(0)
        model.get_logits(s_tokens)
        for layer in layers:
            s_activations = model.get_last_activations(layer)
            s_activations = s_activations[0, -2, :].detach().cpu()
            diffs[layer].append(s_activations)
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -2, :].detach().cpu()
            diffs[layer][-1] -= n_activations
    for layer in layers:
        diffs[layer] = torch.stack(diffs[layer])
        torch.save(diffs[layer], os.path.join(data_path, f"{friendly_name}-all_diffs_layer_{layer}.pt"))
        vec = diffs[layer].mean(dim=0)
        torch.save(vec, os.path.join(data_path, f"{friendly_name}-vec_layer_{layer}.pt"))


def extractDiffs(dataset, checkpoint_root, friendly_name, n_samples=100, n=15 ):
    m2 = friendly_name
    m1 = friendly_name + "_contrastive"

    c2 = checkpoint_root + f"/checkpoint-{n}"
    c1 = checkpoint_root + f"_contrastive/checkpoint-{n}"

    def generate_and_save_steering_vectors(model, dataset, model_name, start_layer=15, end_layer=29):
        layers = list(range(start_layer, end_layer + 1))
        vecs = dict([(layer, []) for layer in layers])
        model.set_save_internal_decodings(False)
        model.reset_all()

        for datum in tqdm( [d for i, d in enumerate(dataset) if i < n_samples] ):
            prompt = datum["input_ids"]
            prompt = prompt.to(model.device)
            index = findLastOccurrence(prompt.unsqueeze(0), assistant_tokens)[0]
            model.get_logits(prompt.unsqueeze(0))
            for layer in layers:
                activations = model.get_last_activations(layer)
                vecs[layer].append(activations[0, index, :].detach().cpu())

        for layer in layers:
            vecs[layer] = torch.stack(vecs[layer])
            torch.save(vecs[layer], os.path.join(data_path, f"{model_name}_vecs_layer_{layer}.pt"))

        return vecs

    zephyr = WrappedZephyr(system_prompt="You are a helpful, honest, and concise assistant.", checkpoint=c2)
    v2 = generate_and_save_steering_vectors(zephyr, dataset, m2, start_layer=13, end_layer=30)

    zephyr = WrappedZephyr(system_prompt="You are a helpful, honest, and concise assistant.", checkpoint=c1)
    v1 = generate_and_save_steering_vectors(zephyr, dataset, m1, start_layer=13, end_layer=30)

    diffs = {i:(v2[i] - v1[i]) for i in range(13, 31)}

    for layer, diff in diffs.items():
        torch.save(diff, os.path.join(data_path, f"{m2}-{m1}_diffs_layer_{layer}.pt"))


def extractActivations(dataset, checkpoint, friendly_name, n_samples=100 ):
    def generate_and_save_steering_vectors(model, dataset, model_name, start_layer=15, end_layer=29):
        layers = list(range(start_layer, end_layer + 1))
        vecs = dict([(layer, []) for layer in layers])
        end_vecs = dict([(layer, []) for layer in layers])
        preds = []
        correct = []
        model.set_save_internal_decodings(False)
        model.reset_all()

        for datum in tqdm( [d for i, d in enumerate(dataset) if i < n_samples] ):
            prompt = datum["input_ids"]

            prompt = prompt.to(model.device)
            index = findLastOccurrence(prompt.unsqueeze(0), assistant_tokens)[0]
            logits = model.get_logits(prompt.unsqueeze(0))[0, -1, :]
            
            if not torch.argmax(logits) in [28741, 28760]:
                continue

            # ridiculous hack which relies on the fact that I processed the Anthropic sycophancy dataset to have labels A and B, which tokenize to 28741 and 28760
            c = 28741 if datum["labels"] == 28741 else 28760
            d = 28741 if datum["labels"] == 28760 else 28760
            if logits[c] > logits[d]:
                pred = c
                correct.append( True )
            else: 
                correct.append( False )
                pred = d

            preds.append(pred)
            for layer in layers:
                activations = model.get_last_activations(layer)
                vecs[layer].append(activations[0, index, :].detach().cpu())
                end_vecs[layer].append(activations[0, -2, :].detach().cpu())

        for layer in layers:
            vecs[layer] = torch.stack(vecs[layer])
            torch.save(vecs[layer], os.path.join(data_path, f"{model_name}_vecs_layer_{layer}.pt"))

            end_vecs[layer] = torch.stack(end_vecs[layer])
            torch.save(end_vecs[layer], os.path.join(data_path, f"{model_name}_end_vecs_layer_{layer}.pt"))

        torch.save(preds, os.path.join(data_path, f"{model_name}_preds.pt"))
        torch.save(correct, os.path.join(data_path, f"{model_name}_correct.pt"))
        return vecs, preds, correct

    zephyr = WrappedZephyr(system_prompt="You are a helpful, honest, and concise assistant.", checkpoint=checkpoint)
    activations, predictions, correct = generate_and_save_steering_vectors(zephyr, dataset, friendly_name, start_layer=13, end_layer=30)
    return activations, predictions, correct


def extractActivations(dataset, checkpoint, friendly_name, n_samples=100 ):
    def generate_and_save_steering_vectors(model, dataset, model_name, start_layer=15, end_layer=29):
        # for datum in tqdm( [d for i, d in enumerate(dataset) if i < n_samples] ):
        #     prompt = datum["prompt"]
        #     prompt = prompt.to(model.device)

        #     tokens = 

        #     continuation = model.generate_text(prompt, max_length=200)


        layers = list(range(start_layer, end_layer + 1))
        vecs = dict([(layer, []) for layer in layers])
        end_vecs = dict([(layer, []) for layer in layers])
        preds = []
        correct = []
        model.set_save_internal_decodings(False)
        model.reset_all()

        tokenizer = buildTokenizer()

        for datum in tqdm( [d for i, d in enumerate(dataset) if i < n_samples] ):
            prompt = datum["input_ids"]

            prompt = prompt.to(model.device)
            index = findLastOccurrence(prompt.unsqueeze(0), assistant_tokens)[0]
            logits = model.get_logits(prompt.unsqueeze(0))[0, -1, :]

            # print( tokenizer.decode(prompt) )
            # print( tokenizer.decode( torch.argmax(logits) ))
            
            # if not torch.argmax(logits) in [28741, 28760]:
            #     continue

            # ridiculous hack which relies on the fact that I processed the Anthropic sycophancy dataset to have labels A and B, which tokenize to 28741 and 28760
            c = 28741 if datum["labels"] == 28741 else 28760
            d = 28741 if datum["labels"] == 28760 else 28760
            if logits[c] > logits[d]:
                pred = c
                correct.append( True )
            else: 
                correct.append( False )
                pred = d

            preds.append(pred)
            for layer in layers:
                activations = model.get_last_activations(layer)
                vecs[layer].append(activations[0, index, :].detach().cpu())
                end_vecs[layer].append(activations[0, -2, :].detach().cpu())

        for layer in layers:
            vecs[layer] = torch.stack(vecs[layer])
            torch.save(vecs[layer], os.path.join(data_path, f"{model_name}_vecs_layer_{layer}.pt"))

            end_vecs[layer] = torch.stack(end_vecs[layer])
            torch.save(end_vecs[layer], os.path.join(data_path, f"{model_name}_end_vecs_layer_{layer}.pt"))

        torch.save(preds, os.path.join(data_path, f"{model_name}_preds.pt"))
        torch.save(correct, os.path.join(data_path, f"{model_name}_correct.pt"))
        return vecs, end_vecs, preds, correct

    zephyr = WrappedZephyr(system_prompt="You are a helpful, honest, and concise assistant.", checkpoint=checkpoint)
    activations, post_activations, predictions, correct = generate_and_save_steering_vectors(zephyr, dataset, friendly_name, start_layer=13, end_layer=30)
    return activations, post_activations, predictions, correct

from utils.data import generate_and_tokenize_prompt
def prePostActivations(dataset, checkpoint, friendly_name, n_samples=100, start_layer=13, end_layer=30, max_length=200 ):
    """
    get the activations after the prompt
    generate a continuation
    get the activations after continuation
    return them all
    """
    tokenizer = buildTokenizer()
    model = WrappedZephyr(system_prompt="You are a helpful, honest, and concise assistant.", checkpoint=checkpoint)
    
    layers = list(range(start_layer, end_layer + 1))
    pre_activations = dict([(layer, []) for layer in layers])
    post_activations = dict([(layer, []) for layer in layers])

    continuations = []
    solutions = []
    for datum in tqdm( [d for i, d in enumerate(dataset) if i < n_samples] ):
        prompt = datum["prompt"]
        correct_answer = datum["solution"]
        solutions.append(correct_answer)
        tokens = generate_and_tokenize_prompt(prompt, tokenizer, add_generation_prompt=True)
        tokens = tokens.to(model.device)

        logits = model.get_logits(tokens.unsqueeze(0))[0, -1, :]
        for layer in layers:
            activations = model.get_last_activations(layer)
            pre_activations[layer].append(activations[0, -1, :].detach().cpu())

        continuation = model.model.generate(tokens.unsqueeze(0), max_length=len(tokens) + max_length)
        continuations.append(tokenizer.decode(continuation[0]))

        logits = model.get_logits(continuation)[0, -2, :]
        for layer in layers:
            activations = model.get_last_activations(layer)
            post_activations[layer].append(activations[0, -2, :].detach().cpu())

    for layer in layers:
        pre_activations[layer] = torch.stack(pre_activations[layer])
        torch.save(pre_activations[layer], os.path.join(data_path, f"{friendly_name}_pre_activations_layer_{layer}.pt"))

        post_activations[layer] = torch.stack(post_activations[layer])
        torch.save(post_activations[layer], os.path.join(data_path, f"{friendly_name}_post_activations_layer_{layer}.pt"))
    
    torch.save(continuations, os.path.join(data_path, f"{friendly_name}_continuations.pt"))
    torch.save(solutions, os.path.join(data_path, f"{friendly_name}_solutions.pt"))

    return pre_activations, post_activations, continuations, solutions