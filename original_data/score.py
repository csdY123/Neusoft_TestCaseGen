import json
import os

with open('/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/chensenda/codes/Neusoft/TestCaseGen/original_data/ai4test_neusoft_eval_data_30_with_pred_evaluated_v3_retry.jsonl', 'r') as f:
    data_list = []
    for line in f:
        data = json.loads(line)
        data_list.append(data)

scores_list = []
for data in data_list:
    print(data["eval_result"]['total_score'])
    scores_list.append(data["eval_result"]['total_score'])

print(scores_list)
print(len(scores_list))
print(sum(scores_list) / len(scores_list))

# Statistics by score ranges
total_count = len(scores_list)
range_8_9 = sum(1 for score in scores_list if 8 <= score <= 9)
range_5_7 = sum(1 for score in scores_list if 5 <= score < 8)
range_0_4 = sum(1 for score in scores_list if 0 <= score < 5)

print("\n=== Score Range Statistics ===")
print(f"8-9 points: {range_8_9} ({range_8_9/total_count*100:.2f}%)")
print(f"5-7 points: {range_5_7} ({range_5_7/total_count*100:.2f}%)")
print(f"0-4 points: {range_0_4} ({range_0_4/total_count*100:.2f}%)")