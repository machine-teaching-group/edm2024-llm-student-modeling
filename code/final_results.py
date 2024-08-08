import json
import os
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from code.utils import models_hoc4, models_hoc18, model_to_method_name, process_annotation_data, compute_success_rate, tgt_tasks_hoc4, tgt_tasks_hoc18, metrics, model_to_bar_idx, patterns
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
mpl.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams.update({'font.size': 50})

def plot(args, hoc4_not_fine_tuned_scores, hoc4_fine_tuned_scores, hoc18_not_fine_tuned_scores, hoc18_fine_tuned_scores, std_hoc4, std_hoc18):
    #====== PLOT ======#
    fig, ax = plt.subplots()
    # hoc4 not fine-tuned models
    for _, (name, metric_value) in enumerate(hoc4_not_fine_tuned_scores.items()):
        if name == 'TutorSS':
            # Draw a horizontal line for TutorSS
            ax.plot([-0.4, 4.6], [metric_value, metric_value], color='red', linestyle='--', linewidth=5)
            ax.text(2.9, metric_value+0.06, '\\textsc{TutorSS}', color='black', fontsize=40, ha='left', va='center')
        else:
            bar_idx = model_to_bar_idx[name]
            bar = ax.bar(bar_idx, metric_value, hatch=patterns[name], color='white', edgecolor='black', linewidth=2, zorder=20)
            bar[0].set_edgecolor('black')

    # hoc4 fine-tuned models
    for _, (name, metric_value) in enumerate(hoc4_fine_tuned_scores.items()):
        bar_idx = model_to_bar_idx[name]
        if 'ft' in name and 'Llama' in name:
            bar = ax.bar(bar_idx, metric_value, color='palegreen', edgecolor='black', linewidth=2, yerr=std_hoc4[name], error_kw=dict(lw=3, capsize=5, capthick=3, ecolor='black', zorder=15))
        else:
            bar = ax.bar(bar_idx, metric_value, color='palegreen', edgecolor='black', linewidth=2)
        bar[0].set_edgecolor('black')

    # hoc18 not fine-tuned models
    for _, (name, metric_value) in enumerate(hoc18_not_fine_tuned_scores.items()):
        if name == 'TutorSS':
            # Draw a horizontal line for TutorSS
            ax.plot([5.6, 10.5], [metric_value, metric_value], color='red', linestyle='--', linewidth=5)
            ax.text(8.8, metric_value+0.06, '\\textsc{TutorSS}', color='black', fontsize=40, ha='left', va='center')
        else:
            bar_idx = model_to_bar_idx[name]
            bar = ax.bar(bar_idx+6, metric_value, hatch=patterns[name], color='white', edgecolor='black', linewidth=2, zorder=10)
            bar[0].set_edgecolor('black')

    # hoc18 fine-tuned models
    for _, (name, metric_value) in enumerate(hoc18_fine_tuned_scores.items()):
        bar_idx = model_to_bar_idx[name]
        if 'ft' in name and 'Llama' in name:
            bar = ax.bar(bar_idx+6, metric_value, color='palegreen', edgecolor='black', linewidth=2, yerr=std_hoc18[name], error_kw=dict(lw=3, capsize=5, capthick=3, ecolor='black', zorder=15))
        else:
            bar = ax.bar(bar_idx+6, metric_value, color='palegreen', edgecolor='black', linewidth=2)
        bar[0].set_edgecolor('black')

    # Plot and save
    idxs_hoc4 = list(range(len(model_to_bar_idx)-1))
    idxs_hoc18 = [x+6 for x in idxs_hoc4]
    idxss = idxs_hoc4 + idxs_hoc18
    labels = ['GPT-4-SS', 'GPT-3.5ft-SS', 'Llama2-7Bft-SS', 'Llama2-70Bft-SS', '\\textsc{NeurSS}',
                'GPT-4-SS', 'GPT-3.5ft-SS', 'Llama2-7Bft-SS', 'Llama2-70Bft-SS', '\\textsc{NeurSS}']
    plt.xticks(idxss, labels, fontsize=45, rotation=45, ha='right', rotation_mode='anchor')
    plt.text(0.8, 1.05, 'HoCMaze-4', fontsize=50, zorder=20)
    plt.text(6.9, 1.05, 'HoCMaze-18', fontsize=50, zorder=20)
    plt.ylabel('\\textsc{' + f'{metrics[metric]}' +'}', fontsize=50)
    plt.ylim(0, 1.2)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, "plots", metrics[metric]+'.svg'), format='svg', dpi=1200)
    plt.savefig(os.path.join(args.output_path, "plots", metrics[metric]+'.pdf'), format='pdf', dpi=1200)
    plt.clf()


def compute_avg(metric, renamed_data_hoc4, renamed_data_hoc18):
    success_rate_hoc4 = compute_success_rate(renamed_data_hoc4, models_list=models_hoc4, tgt_task=tgt_tasks_hoc4, metric=metric)
    success_rate_hoc18 = compute_success_rate(renamed_data_hoc18, models_list=models_hoc18, tgt_task=tgt_tasks_hoc18, metric=metric)
    
    avg_success_rate_hoc4 = {}
    avg_success_rate_hoc18 = {}
    std_hoc4 = {}
    std_hoc18 = {}
    hoc4_not_fine_tuned_scores = {}
    hoc4_fine_tuned_scores = {}
    hoc18_not_fine_tuned_scores = {}
    hoc18_fine_tuned_scores = {}

    #====== HOC4 ======#
    avg_success_rate_hoc4['GPT-4'] = success_rate_hoc4['GPT-4']
    avg_success_rate_hoc4['chatGPT-GPT-3.5'] = success_rate_hoc4['chatGPT-GPT-3.5']
    avg_success_rate_hoc4['GPT-3.5-turbo-0613-fine-tuned-5-epochs'] = success_rate_hoc4['GPT-3.5-turbo-0613-fine-tuned-5-epochs-seed-0']
    avg_success_rate_hoc4['llama-2-7b-chat'] = success_rate_hoc4['llama-2-7b-chat']
    avg_success_rate_hoc4['llama-2-70b-chat'] = success_rate_hoc4['llama-2-70b-chat']
    avg_success_rate_hoc4['llama-2-7b-chat-fine-tuned-2-epochs'] = (success_rate_hoc4['llama-2-7b-chat-fine-tuned-2-epochs-seed-0'] + success_rate_hoc4['llama-2-7b-chat-fine-tuned-2-epochs-seed-1'] + success_rate_hoc4['llama-2-7b-chat-fine-tuned-2-epochs-seed-2'])/3
    avg_success_rate_hoc4['llama-2-70b-chat-fine-tuned-2-epochs'] = (success_rate_hoc4['llama-2-70b-chat-fine-tuned-2-epochs-seed-0'] + success_rate_hoc4['llama-2-70b-chat-fine-tuned-2-epochs-seed-1'] + success_rate_hoc4['llama-2-70b-chat-fine-tuned-2-epochs-seed-2'])/3
    avg_success_rate_hoc4['neurSS'] = success_rate_hoc4['neurSS']
    avg_success_rate_hoc4['tutorSS'] = (success_rate_hoc4['tutorSS_1'] + success_rate_hoc4['tutorSS_2'] + success_rate_hoc4['tutorSS_3'])/3

    # Compute standard deviation
    for model_name in avg_success_rate_hoc4:
        if model_name == 'llama-2-7b-chat-fine-tuned-2-epochs':
            std_hoc4[model_to_method_name[model_name]] = np.std([success_rate_hoc4['llama-2-7b-chat-fine-tuned-2-epochs-seed-0'], success_rate_hoc4['llama-2-7b-chat-fine-tuned-2-epochs-seed-1'], success_rate_hoc4['llama-2-7b-chat-fine-tuned-2-epochs-seed-2']])
        elif model_name == 'llama-2-70b-chat-fine-tuned-2-epochs':
            std_hoc4[model_to_method_name[model_name]] = np.std([success_rate_hoc4['llama-2-70b-chat-fine-tuned-2-epochs-seed-0'], success_rate_hoc4['llama-2-70b-chat-fine-tuned-2-epochs-seed-1'], success_rate_hoc4['llama-2-70b-chat-fine-tuned-2-epochs-seed-2']])        
        else:
            std_hoc4[model_to_method_name[model_name]] = 0
        # Separate fine-tuned and not fine-tuned models
        if 'fine-tuned' in model_name:
            hoc4_fine_tuned_scores[model_to_method_name[model_name]] = avg_success_rate_hoc4[model_name]
        else:
            hoc4_not_fine_tuned_scores[model_to_method_name[model_name]] = avg_success_rate_hoc4[model_name]

    #====== HOC18 ======#
    avg_success_rate_hoc18['GPT-4'] = success_rate_hoc18['GPT-4']
    avg_success_rate_hoc18['chatGPT-GPT-3.5'] = success_rate_hoc18['chatGPT-GPT-3.5']
    avg_success_rate_hoc18['GPT-3.5-turbo-0613-fine-tuned-2-epochs'] = success_rate_hoc18['GPT-3.5-turbo-0613-fine-tuned-2-epochs-seed-0']
    avg_success_rate_hoc18['llama-2-7b-chat'] = success_rate_hoc18['llama-2-7b-chat']
    avg_success_rate_hoc18['llama-2-70b-chat'] = success_rate_hoc18['llama-2-70b-chat']
    avg_success_rate_hoc18['llama-2-7b-chat-fine-tuned-10-epochs'] = (success_rate_hoc18['llama-2-7b-chat-fine-tuned-10-epochs-seed-0'] + success_rate_hoc18['llama-2-7b-chat-fine-tuned-10-epochs-seed-1'] + success_rate_hoc18['llama-2-7b-chat-fine-tuned-10-epochs-seed-2'])/3
    avg_success_rate_hoc18['llama-2-70b-chat-fine-tuned-10-epochs'] = (success_rate_hoc18['llama-2-70b-chat-fine-tuned-10-epochs-seed-0'] + success_rate_hoc18['llama-2-70b-chat-fine-tuned-10-epochs-seed-1'] + success_rate_hoc18['llama-2-70b-chat-fine-tuned-10-epochs-seed-2'])/3    
    avg_success_rate_hoc18['neurSS'] = success_rate_hoc18['neurSS']
    avg_success_rate_hoc18['tutorSS'] = (success_rate_hoc18['tutorSS_1'] + success_rate_hoc18['tutorSS_2'] + success_rate_hoc18['tutorSS_3'])/3

    # Compute standard deviation
    for model_name in avg_success_rate_hoc18:
        if model_name == 'llama-2-7b-chat-fine-tuned-10-epochs':
            std_hoc18[model_to_method_name[model_name]] = np.std([success_rate_hoc18['llama-2-7b-chat-fine-tuned-10-epochs-seed-0'], success_rate_hoc18['llama-2-7b-chat-fine-tuned-10-epochs-seed-1'], success_rate_hoc18['llama-2-7b-chat-fine-tuned-10-epochs-seed-2']])
        elif model_name == 'llama-2-70b-chat-fine-tuned-10-epochs':
            std_hoc18[model_to_method_name[model_name]] = np.std([success_rate_hoc18['llama-2-70b-chat-fine-tuned-10-epochs-seed-0'], success_rate_hoc18['llama-2-70b-chat-fine-tuned-10-epochs-seed-1'], success_rate_hoc18['llama-2-70b-chat-fine-tuned-10-epochs-seed-2']])
        elif model_name == 'GPT-3.5-turbo-0613-fine-tuned-2-epochs':
            std_hoc18[model_to_method_name[model_name]] = 0
        else:
            std_hoc18[model_to_method_name[model_name]] = 0
        # Separate fine-tuned and not fine-tuned models
        if 'fine-tuned' in model_name:
            hoc18_fine_tuned_scores[model_to_method_name[model_name]] = avg_success_rate_hoc18[model_name]
        else:
            hoc18_not_fine_tuned_scores[model_to_method_name[model_name]] = avg_success_rate_hoc18[model_name]

    return hoc4_not_fine_tuned_scores, hoc4_fine_tuned_scores, hoc18_not_fine_tuned_scores, hoc18_fine_tuned_scores, std_hoc4, std_hoc18
    

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_data_path', type=str, default='data/expert_annotations/fine_grained_annotations/')
    parser.add_argument('--output_path', type=str, default='outputs/hoc')
    args = parser.parse_args()
    
    renamed_data_hoc4, renamed_data_hoc18 = process_annotation_data(data_path=args.annotation_data_path)
    
    for metric in metrics.keys():
        hoc4_not_fine_tuned_scores, hoc4_fine_tuned_scores, hoc18_not_fine_tuned_scores, hoc18_fine_tuned_scores, std_hoc4, std_hoc18 = compute_avg(metric, renamed_data_hoc4, renamed_data_hoc18)

        plot(args, hoc4_not_fine_tuned_scores, hoc4_fine_tuned_scores, hoc18_not_fine_tuned_scores, hoc18_fine_tuned_scores, std_hoc4, std_hoc18)
        


