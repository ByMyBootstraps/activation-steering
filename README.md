# Intro
This project attempts to use activation diffs to predict whether a model is going to produce an output with a given feature, as well as to retroactively predict whether a completed output has that feature.

This project investigates two questions:
1) can we use steering vectors to predict whether an output will be / was sycophantic?
2) how much does predictivity vary depending on how we generate the steering vectors?

To that end, I:
a) generate steering vectors in multiple ways and explore how they vary in subjective effect on model outputs and cosine similarity
b) measure how much the cosine similarity between the model's actual activations and the steering vectors correlates with the sycophantism of the output
(except (a) is largely a work in progress. (b) is done though).

This project is very unfinished as of 11/17/23 because I wanted to publish this codebase so I could link in on my MATS application.
In particular, all experiments are performed on Zephyr-7B beta at very small scale and I really want to acquire better datasets.

Key differences from [Reducing sycophancy and improving honesty via activation steering](https://www.lesswrong.com/posts/zt6hRsDE84HeBKh7E/reducing-sycophancy-and-improving-honesty-via-activation#Reducing_sycophancy_via_activation_steering):
- I mostly used finetuning to get my diffs, more on that below
- Zephyr is heavily fine-tuned

# Predictivity of steering vectors
Premise: if we have successfully identified a vector that encodes something like "respond sycophantically", we might expect that, in situations were the model is spontaneously behaving sycophantically, its activations would have high cosine similarity with that vector.

I perform two tests. 
One: For each prompt in a subset of Anthropic's [sycophancy dataset](https://huggingface.co/datasets/Anthropic/model-written-evals/tree/main/sycophancy), I measure whether the model chooses the sycophantic option response and observe its activation immediately after that prompt to see if I can predict whether it will respond sycophantically.

Two: For each prompt in a subset of the sycophancy dataset, I present the model with a transcript where the decision has already been made and attempt to infer from its activation whether that decision was the sycophantic option.

I also tried this with a subset of TruthfulQA with answers which I graded by hand. 

In both cases: the steering vector was highly predictive of sycophancy in distribution (the examples are also from the sycophancy dataset) and not at all predictive out of distribution (on TruthfulQA). However, this comes with the important caveat that my out of distribution test was pretty bad.

Additionally, the steering vector was not at all predictive on layer 13. The predictivity required that we look at layer 30.

![truthful qa with continuation 13](./images/truthfulqa_continuation_13.png)
![truthful qa from prompt 13](./images/truthfulqa_prompt_13.png)

![truthful qa with continuation 30](./images/truthfulqa_continuation_30.png)
![truthful qa from prompt 30](./images/truthfulqa_prompt_30.png)


![anthropic with continuation 13](./images/anthropic_continuation_13.png)
![anthropic from prompt 13](./images/anthropic_prompt_13.png)

![anthropic with continuation 30](./images/anthropic_continuation_30.png)
![anthropic from prompt 30](./images/anthropic_prompt_30.png)
