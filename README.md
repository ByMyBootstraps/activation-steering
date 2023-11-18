# Intro
The original post [Reducing sycophancy and improving honesty via activation steering](https://www.lesswrong.com/posts/zt6hRsDE84HeBKh7E/reducing-sycophancy-and-improving-honesty-via-activation#Reducing_sycophancy_via_activation_steering) identified vectors which could be used to induce or reduce sycophancy in LLaMA-7B by adding them to the model's activations during inference. The implication was that the vector 
This project attempts to use activation diffs to predict whether a model is going to produce an output with a given feature, as well as to retroactively predict whether a completed output has that feature.

The author finds that an activation vector derived by contrasting sycophantic and non-sycophantic activations on Anthropic's [sycophancy dataset](https://huggingface.co/datasets/Anthropic/model-written-evals/tree/main/sycophancy) can be used to influence whether the model reproduces common myths and misconceptions on [TruthfulQA](https://huggingface.co/datasets/truthful_qa) and concludes that there is a shared direction between sycophancy on questions of opinion and endorsing common false beliefs on questions of fact.

This project aims to investigate two questions:
1) can we use steering vectors to predict sycophancy rather than inducing it?
2) how much does predictivity vary depending on how we generate the steering vectors? [WIP]

The observation that a single activation steering vector can be used to modulate the model's dishonesty in the two contexts doesn't necessarily mean that it encodes anything to do with dishonesty, per se. It could encode some other thing that incidentally leads to dishonesty such as, say, attending heavily to previous word choice.

If steering vectors are robustly predictive, that suggests that, in practice, sycophancy tends to arise for the same reason in diverse situations (namely because its intermediate activations pointed towards that direction).
That would suggest that they represent something closer to a general optimization target than a specific structural feature which happens to lead to sycophancy.

To that end, I:
a) generate steering vectors in multiple ways and explore how they vary in usefulness for steering the model and for prediciting model behavior
b) measure how much the cosine similarity between the model's actual activations and the steering vectors correlates with the sycophantism of the output
(except (a) is largely a work in progress).

This project is very unfinished as of 11/17/23 because I wanted to publish this codebase so I could link in on my MATS application.
In particular, all experiments are performed on LLaMa or Zephyr-7B beta at very small scale and I really want to acquire more diverse datasets and reproduce some of the original author's later experiments. Additionally, I don't yet answer the questions which originally drove this project: how much do extracted steering vectors vary depending on the set of prompts they were extracted from? And what features of the extraction process influence the predictivity of the resulting vectors?

Note that, unlike, the original paper, I do my experiments on Zephyr-7b-beta, which is heavily fine-tuned for chat and honesty.

# Predictivity of steering vectors
Premise: if we have successfully identified a vector that encodes something like "respond sycophantically", we might expect that, in situations were the model is spontaneously behaving sycophantically, its activations would have high cosine similarity with that vector. Whereas if it encodes some other target which leads to sycophancy there is less reason to expect unrelated sycophancy to have the same cause.

I extract steering vectors for sycophancy from activations on Anthropic's [sycophancy dataset](https://huggingface.co/datasets/Anthropic/model-written-evals/tree/main/sycophancy) in the same fashion as the original paper.

Then, for each prompt in a different subset, I generate model continuations and record whether the continuation was sycophantic, the model's intermediate activations after being given the prompt, and the model's intermediate activations after generating its continuation.

Then I test whether the cosine similarity between either of those activations and my steering vectors is predictive of sycophancy on other examples.

I also tried this with a subset of TruthfulQA with answers which I graded by hand. 

For both post-prompt and post-continuation activations: the steering vector was highly predictive of sycophancy in distribution (the examples are also from the sycophancy dataset) and not at all predictive out of distribution (on TruthfulQA). However, this comes with the important caveat that my out of distribution test was pretty bad.

Confusingly, sycophancy correlates positively with cosine similarity at layer 13 and then smoothly decreases up to layer 30 where it is significantly negative.

## TruthfulQA Cosine Similarity
![truthful qa with continuation 13](./images/Zephyr_TruthfulQA_continuation_13.png)
![truthful qa from prompt 13](./images/Zephyr_TruthfulQA_prompt_13.png)

![truthful qa with continuation 30](./images/Zephyr_TruthfulQA_continuation_30.png)
![truthful qa from prompt 30](./images/Zephyr_TruthfulQA_prompt_30.png)

## Anthropic Sycophancy Cosine Similarity
### Zephyr
![anthropic with continuation 13](./images/Zephyr_Anthropic_continuation_13.png)
![anthropic from prompt 13](./images/Zephyr_Anthropic_prompt_13.png)

![anthropic with continuation 30](./images/Zephyr_Anthropic_continuation_30.png)
![anthropic from prompt 30](./images/Zephyr_Anthropic_prompt_30.png)

### LLaMA
*Note*: these graphs use the steering vector derived from LLaMA-7B to predict the sycophancy of Zhepyr-7B. This is because LLaMA-7B is not instruction tuned and didn't obey the target output format and I have not implemented a system for labeling LLaMA-7B outputs as sycophantic or not.

![anthropic with continuation 13](./images/LLaMA_Anthropic_continuation_13.png)
![anthropic from prompt 13](./images/LLaMA_Anthropic_prompt_13.png)

![anthropic with continuation 30](./images/LLaMA_Anthropic_continuation_30.png)
![anthropic from prompt 30](./images/LLaMA_Anthropic_prompt_30.png)

# Conclusions
Steering vectors derived from both LLaMA-7B and from Zephyr-7B can be used to predict both whether Zephyr-7B is going to be sycophantic and whether a generated response was sycophantic *if* both the steering vectors and the tests use prompts drawn from Anthropic's [sycophancy dataset](https://huggingface.co/datasets/Anthropic/model-written-evals/tree/main/sycophancy).

This is mildly suggestive that they could be predictive in general but leaves much work to be done.

# Next steps
The original conceit of this project was to investigate whether fine-tuning a highly steerable model like Zephyr for sycophancy would lead it to do something like activation steering on itself; specifically, can we extract steering vectors by comparing the intermediate activations of sycophantic!zephyr with those of unsycophantic!zephyr on a fixed prompt?

This would provide hints about how foundation models learn new skills and, if answered posivitely, would cause me to update towards believing that large language models can meaningfully be said to have simple encodings for "behave sycophantically".