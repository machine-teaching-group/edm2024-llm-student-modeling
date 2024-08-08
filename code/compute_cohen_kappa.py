import json
import os
import argparse

def rename_keys(original_data, task_map, method_map, student_map):
    '''
        The data was masked to hide the source method, task, and student. This function reverses the masking.
    '''
    renamed_data = {}
    for tgt_task, tgt_value in original_data.items():
        new_tgt_task = task_map[tgt_task]
        renamed_data[new_tgt_task] = {}
        for stu, stu_value in tgt_value.items():
            new_stu = student_map[stu]
            renamed_data[new_tgt_task][new_stu] = {}
            for method, method_value in stu_value.items():
                if method == 'mis':
                    continue
                new_method = method_map[method]
                renamed_data[new_tgt_task][new_stu][new_method] = method_value
    return renamed_data

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def process_expert_data(expert_data_path, expert_num):
    expert_hoc4 = load_json(os.path.join(expert_data_path, f'{expert_num}_expert/results_hoc4.json'))
    expert_hoc18 = load_json(os.path.join(expert_data_path, f'{expert_num}_expert/results_hoc18.json'))

    student_map = load_json(os.path.join(expert_data_path, f'{expert_num}_expert/name_mapping/student_map.json'))
    hoc4_task_map = load_json(os.path.join(expert_data_path, f'{expert_num}_expert/name_mapping/hoc4_task_map.json'))
    hoc18_task_map = load_json(os.path.join(expert_data_path, f'{expert_num}_expert/name_mapping/hoc18_task_map.json'))
    hoc4_method_map = load_json(os.path.join(expert_data_path, f'{expert_num}_expert/name_mapping/hoc4_method_map.json'))
    hoc18_method_map = load_json(os.path.join(expert_data_path, f'{expert_num}_expert/name_mapping/hoc18_method_map.json'))

    renamed_expert_hoc4 = rename_keys(expert_hoc4, hoc4_task_map, hoc4_method_map, student_map)
    renamed_expert_hoc18 = rename_keys(expert_hoc18, hoc18_task_map, hoc18_method_map, student_map)

    return renamed_expert_hoc4, renamed_expert_hoc18

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_data_path', type=str, default='data/expert_annotations/binary_annotations')
    args = parser.parse_args()

    renamed_first_expert_hoc4, renamed_first_expert_hoc18 = process_expert_data(args.annotation_data_path, '1st')
    renamed_second_expert_hoc4, renamed_second_expert_hoc18 = process_expert_data(args.annotation_data_path, '2nd')

    i0_j0, i0_j1, i1_j0, i1_j1 = 0, 0, 0, 0

    expert_pairs = [(renamed_first_expert_hoc4, renamed_second_expert_hoc4), (renamed_first_expert_hoc18, renamed_second_expert_hoc18)]

    # Compute Cohen's Kappa
    for first_expert, second_expert in expert_pairs:
        for tgt_task, tgt_value in second_expert.items():
            for stu, stu_value in tgt_value.items():
                for method, method_value in stu_value.items():
                    if method == 'mis':
                        continue
                    first_expert_value = first_expert.get(tgt_task, {}).get(stu, {}).get(method, {}).get('both_stu_and_task', 0)
                    second_expert_value = second_expert.get(tgt_task, {}).get(stu, {}).get(method, {}).get('both_stu_and_task', 0)

                    if first_expert_value == 1 and second_expert_value == 1:
                        i0_j0 += 1
                    elif first_expert_value == 1 and second_expert_value == 0:
                        i0_j1 += 1
                    elif first_expert_value == 0 and second_expert_value == 1:
                        i1_j0 += 1
                    elif first_expert_value == 0 and second_expert_value == 0:
                        i1_j1 += 1
    
    total = i0_j0 + i0_j1 + i1_j0 + i1_j1
    po = (i0_j0 + i1_j1) / total
    pe = ((i0_j0 + i0_j1) * (i0_j0 + i1_j0) + (i1_j0 + i1_j1) * (i0_j1 + i1_j1)) / (total * total)
    kappa = (po - pe) / (1 - pe)
    print("Cohen's Kappa: ", kappa)
