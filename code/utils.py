import os
import json

models_hoc4 = ['chatGPT-GPT-3.5', 
                'GPT-4',
                'GPT-3.5-turbo-0613-fine-tuned-5-epochs-seed-0',
                'llama-2-7b-chat',
                'llama-2-70b-chat',
                'llama-2-7b-chat-fine-tuned-2-epochs-seed-0',
                'llama-2-7b-chat-fine-tuned-2-epochs-seed-1',
                'llama-2-7b-chat-fine-tuned-2-epochs-seed-2',
                'llama-2-70b-chat-fine-tuned-2-epochs-seed-0',
                'llama-2-70b-chat-fine-tuned-2-epochs-seed-1',
                'llama-2-70b-chat-fine-tuned-2-epochs-seed-2',
                'neurSS',
                'tutorSS_1',
                'tutorSS_2',
                'tutorSS_3',
                ]

models_hoc18 = ['chatGPT-GPT-3.5', 
                'GPT-4',
                'GPT-3.5-turbo-0613-fine-tuned-2-epochs-seed-0',
                'llama-2-7b-chat',
                'llama-2-70b-chat',
                'llama-2-7b-chat-fine-tuned-10-epochs-seed-0',
                'llama-2-7b-chat-fine-tuned-10-epochs-seed-1',
                'llama-2-7b-chat-fine-tuned-10-epochs-seed-2',
                'llama-2-70b-chat-fine-tuned-10-epochs-seed-0',
                'llama-2-70b-chat-fine-tuned-10-epochs-seed-1',
                'llama-2-70b-chat-fine-tuned-10-epochs-seed-2',
                'neurSS',
                'tutorSS_1',
                'tutorSS_2',
                'tutorSS_3',
                ]

model_to_method_name = {'chatGPT-GPT-3.5':'GPT-3.5ft-SS', 
              'GPT-4':'GPT-4-SS',
              'GPT-3.5-turbo-0613-fine-tuned-5-epochs':'GPT-3.5ft-SS', 
              'GPT-3.5-turbo-0613-fine-tuned-2-epochs':'GPT-3.5ft-SS', 
              'llama-2-7b-chat':'Llama2-7Bft-SS',
              'llama-2-70b-chat':'Llama2-70Bft-SS',
              'llama-2-7b-chat-fine-tuned-2-epochs':'Llama2-7Bft-SS',
              'llama-2-7b-chat-fine-tuned-10-epochs':'Llama2-7Bft-SS',
              'llama-2-70b-chat-fine-tuned-2-epochs':'Llama2-70Bft-SS',
              'llama-2-70b-chat-fine-tuned-10-epochs':'Llama2-70Bft-SS',
              'neurSS':'NeurSS',
              'tutorSS':'TutorSS'}

stu_list = ['stu'+str(i) for i in range(1, 7)]
tgt_tasks_hoc4 = ['hoc4a', 'hoc4b', 'hoc4c']
tgt_tasks_hoc18 = ['hoc18a', 'hoc18b', 'hoc18c']


metrics = {"stu_behavior":"q-stu",
            "task_characteristic":"q-task",
            "both_stu_and_task":"q-overall"}

model_to_bar_idx = {'GPT-4-SS':0,
                    'GPT-3.5ft-SS':1,
                    'Llama2-7Bft-SS':2,
                    'Llama2-70Bft-SS':3, 
                    'NeurSS':4,
                    'TutorSS':5}
        
patterns = {'ChatGPT-3.5-SS':"-", 
    'GPT-4-SS':"/",
    'GPT-3.5ft-SS':"+", 
    'Llama2-7B-SS':"*",
    'Llama2-70B-SS':"",
    'Llama2-7Bft-SS':"o",
    'Llama2-70Bft-SS':"|",  
    'NeurSS':".",
    'TutorSS':""}

def process_annotation_data(data_path):
    with open(os.path.join(data_path, 'results_hoc4.json'), 'r') as f:
        # Load the JSON data_hoc4
        data_hoc4 = json.load(f)
    
    with open(os.path.join(data_path, 'results_hoc18.json'), 'r') as f:
        # Load the JSON data_hoc18
        data_hoc18 = json.load(f)

    with open(os.path.join(data_path, 'name_mapping/student_map.json'), 'r') as f:
        student_map = json.load(f)
    with open(os.path.join(data_path, 'name_mapping/hoc4_task_map.json'), 'r') as f:
        hoc4_task_map = json.load(f)
    with open(os.path.join(data_path, 'name_mapping/hoc18_task_map.json'), 'r') as f:
        hoc18_task_map = json.load(f)
    with open(os.path.join(data_path, 'name_mapping/hoc4_method_map.json'), 'r') as f:
        hoc4_method_map = json.load(f)
    with open(os.path.join(data_path, 'name_mapping/hoc18_method_map.json'), 'r') as f:
        hoc18_method_map = json.load(f)
    
    renamed_data_hoc4 = {}
    for tgt_task, tgt_value in data_hoc4.items():
        new_tgt_task = hoc4_task_map[tgt_task]
        renamed_data_hoc4[new_tgt_task] = {}
        for stu, stu_value in tgt_value.items():
            new_stu = student_map[stu]
            renamed_data_hoc4[new_tgt_task][new_stu] = {}
            for method, method_value in stu_value.items():
                if method == 'mis':
                    continue
                new_method = hoc4_method_map[method]        
                renamed_data_hoc4[new_tgt_task][new_stu][new_method] = method_value

    renamed_data_hoc18 = {}
    for tgt_task, tgt_value in data_hoc18.items():
        new_tgt_task = hoc18_task_map[tgt_task]
        renamed_data_hoc18[new_tgt_task] = {}
        for stu, stu_value in tgt_value.items():
            new_stu = student_map[stu]
            renamed_data_hoc18[new_tgt_task][new_stu] = {}
            for method, method_value in stu_value.items():
                if method == 'mis':
                    continue
                new_method = hoc18_method_map[method]        
                renamed_data_hoc18[new_tgt_task][new_stu][new_method] = method_value

    return renamed_data_hoc4, renamed_data_hoc18

def compute_success_rate(renamed_data, models_list, tgt_task, metric):
    # for metric in metrics.keys():
    metric_sum = {}
    for model_name in models_list:
        metric_sum[model_name] = 0
        for task in tgt_task:
            for stu_id in stu_list:
                try:
                    if metric=='both_stu_and_task':
                        metric_sum[model_name] += \
                            renamed_data[task][stu_id][model_name]['stu_behavior'] *\
                            renamed_data[task][stu_id][model_name]['task_characteristic'] 
                    else:
                        metric_sum[model_name] += renamed_data[task][stu_id][model_name][metric]
                # Print exception
                except Exception as e:
                    print("Exception: ", e)
                    print("model_name: ", model_name)
                    print("task: ", task)
                    print("stu_id: ", stu_id)
                    print("metric: ", metric)
    for model in metric_sum:
        metric_sum[model] = metric_sum[model]/(len(tgt_task)*len(stu_list))
    return metric_sum