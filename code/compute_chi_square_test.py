from code.utils import compute_success_rate, models_hoc4, models_hoc18, model_to_method_name, process_annotation_data, tgt_tasks_hoc4, tgt_tasks_hoc18, stu_list, metrics
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency
import numpy as np
import argparse
import os

# main function
if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_data_path', type=str, default='data/expert_annotations/annotations')
    parser.add_argument('--output_path', type=str, default='outputs/hoc/chisquare_test')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Process the annotation data
    expert_1_renamed_data_hoc4, expert_1_renamed_data_hoc18 = process_annotation_data(data_path= args.annotation_data_path)

    # Compute the count of each q_overall value for each model in hoc4
    count_df = pd.DataFrame(columns=['model_name', '0', '0.25', '0.5', '1'])
    int_to_idx = {0: '0', 0.25: '0.25', 0.5:'0.5', 1:'1'}

    for model_name in models_hoc4:
        # add a row for each model
        count_df = count_df.append({'model_name': model_name, '0': 0, '0.25': 0, '0.5': 0, '1': 0}, ignore_index=True)
        for task in tgt_tasks_hoc4:
            for stu_id in stu_list:
                # compute the value of q_overall
                q_overall = expert_1_renamed_data_hoc4[task][stu_id][model_name]['stu_behavior'] *\
                            expert_1_renamed_data_hoc4[task][stu_id][model_name]['task_characteristic'] 
                count_df.loc[count_df['model_name'] == model_name, int_to_idx[q_overall]] += 1

    # Compute the count of each q_overall value for each model in hoc18
    for model_name in models_hoc18:
        if model_name not in count_df['model_name'].values:
            count_df = count_df.append({'model_name': model_name, '0': 0, '0.25': 0, '0.5': 0, '1': 0}, ignore_index=True)
        for task in tgt_tasks_hoc18:
            for stu_id in stu_list:
                # compute the value of q_overall
                q_overall = expert_1_renamed_data_hoc18[task][stu_id][model_name]['stu_behavior'] *\
                            expert_1_renamed_data_hoc18[task][stu_id][model_name]['task_characteristic'] 
                count_df.loc[count_df['model_name'] == model_name, int_to_idx[q_overall]] += 1

    # Compute the sum for GPT-3.5-turbo-0613-fine-tuned in two reference tasks
    count_df = count_df.append({'model_name': 'GPT-3.5ft-SS', '0': 0, '0.25': 0, '0.5': 0, '1': 0}, ignore_index=True)
    count_df.loc[count_df['model_name'] == 'GPT-3.5ft-SS', ['0', '0.25', '0.5', '1']] = count_df.loc[count_df['model_name'] == 'GPT-3.5-turbo-0613-fine-tuned-5-epochs-seed-0', ['0', '0.25', '0.5', '1']].values[0] + count_df.loc[count_df['model_name'] == 'GPT-3.5-turbo-0613-fine-tuned-2-epochs-seed-0', ['0', '0.25', '0.5', '1']].values[0]
    count_df = count_df.drop(count_df[count_df['model_name'].str.contains('GPT-3.5-turbo-0613-fine-tuned')].index)

    # Compute the average of the three seeds for llama-2-7b
    count_df = count_df.append({'model_name': 'Llama2-7Bft-SS', '0': 0, '0.25': 0, '0.5': 0, '1': 0}, ignore_index=True)
    count_df.loc[count_df['model_name'] == 'Llama2-7Bft-SS', ['0', '0.25', '0.5', '1']] = (count_df[count_df['model_name'].str.contains('llama-2-7b-chat-fine-tuned')][['0', '0.25', '0.5', '1']].sum()/3).values
    count_df = count_df.drop(count_df[count_df['model_name'].str.contains('llama-2-7b-chat-fine-tuned')].index)

    # Compute the average of the three seeds for llama-2-70b-chat-fine-tuned
    count_df = count_df.append({'model_name': 'Llama2-70Bft-SS', '0': 0, '0.25': 0, '0.5': 0, '1': 0}, ignore_index=True)
    count_df.loc[count_df['model_name'] == 'Llama2-70Bft-SS', ['0', '0.25', '0.5', '1']] = (count_df[count_df['model_name'].str.contains('llama-2-70b-chat-fine-tuned')][['0', '0.25', '0.5', '1']].sum()/3).values
    count_df = count_df.drop(count_df[count_df['model_name'].str.contains('llama-2-70b-chat-fine-tuned')].index)

    # Compute the average of the three seeds for TutorSS
    count_df = count_df.append({'model_name': 'TutorSS', '0': 0, '0.25': 0, '0.5': 0, '1': 0}, ignore_index=True)
    count_df.loc[count_df['model_name'] == 'TutorSS', ['0', '0.25', '0.5', '1']] = (count_df[count_df['model_name'].str.contains('tutorSS')][['0', '0.25', '0.5', '1']].sum()/3).values
    count_df = count_df.drop(count_df[count_df['model_name'].str.contains('tutorSS')].index)

    # reset index
    count_df = count_df.reset_index(drop=True)

    # Save the result to a text file 
    with open(args.output_path + '/chisquare_test.txt', 'w') as f: 
        # For each pair of models, compute the chi-squared test statistic and p-value
        for i in range(len(count_df)):
            for j in range(i+1, len(count_df)):
                print(count_df['model_name'][i], count_df['model_name'][j])
                print(chi2_contingency([count_df.iloc[i][['0', '0.25', '0.5', '1']], count_df.iloc[j][['0', '0.25', '0.5', '1']]]))
                # Write the result to the text file
                f.write(count_df['model_name'][i] + ' ' + count_df['model_name'][j] + '\n')
                f.write(str(chi2_contingency([count_df.iloc[i][['0', '0.25', '0.5', '1']], count_df.iloc[j][['0', '0.25', '0.5', '1']]])) + '\n')


   
