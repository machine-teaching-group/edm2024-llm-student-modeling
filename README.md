## [EDM 2024] Large Language Models for In-Context Student Modeling: Synthesizing Student’s Behavior in Visual Programming
This repository contains the implementation for the paper [Large Language Models for In-Context Student Modeling: Synthesizing Student’s Behavior in Visual Programming](https://educationaldatamining.org/edm2024/proceedings/2024.EDM-short-papers.31/2024.EDM-short-papers.31.pdf).

----------------------------------------
### Overview

The repository has the following structure:
* `code/`: this folder contains files required for the evaluation of our framework LLM-SS on the benchmark StudentSyn.
* `data/`: this folder contains data necessary for the evaluation.
* `outputs/`: this is the output folder where the evaluation results will be stored.
* `outputs/hoc/checkpoints`: this is the folder where fine-tuned model checkpoints are stored.

Required packages can be installed by running `pip install -r requirements.txt`.

----------------------------------------
### Compute and visualize the final results
* `python -m code.final_results --annotation_data_path data/expert_annotations/fine_grained_annotations --output_path outputs/hoc`

----------------------------------------
### Compute Cohen Kappa aggreement and Chi-square test
* `python -m code.compute_cohen_kappa --annotation_data_path data/expert_annotations/binary_annotations`
* `python -m code.compute_chi_square_test --annotation_data_path data/expert_annotations/fine_grained_annotations --output_path outputs/hoc/chisquare_test`

----------------------------------------
### Plot fine-tuning loss and BLEU curves
* `python -m code.plot_finetuning_stats --data_path data/finetuning_stats --output_path outputs/hoc/plots`
