from ...smp import *


# def build_sanguo_prompt(line):
#     question = line['question']
#     gt = str(line['answer'])
#     prediction = str(line['prediction'])

#     prompt = """
# Compare the ground truth and prediction from AI models, to give a correctness score for the prediction.
# <AND> in the ground truth means it is totally right
# only when all elements in the ground truth are present in the prediction,
# and <OR> means it is totally right when any one element in the ground truth is present in the prediction.
# The correctness score is 0.0 (totally wrong), 0.1, ..., or 1.0 (totally right).
# Just complete the last space of the correctness score.

# Question | Ground truth | Prediction | Correctness
# --- | --- | --- | ---
# Who is this person? | Guan Yu <OR> Lord Guan | It's Liu Bei | 0.0
# Who is this person? | Guan Yu <OR> Lord Guan | Lord Guan | 1.0
# Why did the fire start? | Zhou Yu set the fire <AND> to burn Cao Cao's fleet | A fire started during naval battle | 0.2
# ...
# """
#     return prompt + '\n' + ' | '.join(
#         [question, gt.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> '), prediction, ''])

def build_sanguo_prompt(line):
    question = line['question']
    gt = str(line['answer'])    # gt - ground truth
    prediction = str(line['prediction'])

    prompt = f"""
Question: {question}
Ground truth: {gt}
Prediction: {prediction}

<和> means all parts of the answer must appear in the prediction.
<或> means any one part is enough.

Give a score between 0.0 (completely wrong) and 1.0 (completely correct), in steps of 0.1.
Only return the number as the final score.
Score:
"""
    return prompt



def Sanguo_auxeval(model, line):
    def float_cvt(s):
        try:
            return float(s)
        except ValueError:
            return None

    prompt = build_sanguo_prompt(line)
    log = ''
    for i in range(5):
        output = model.generate(prompt, temperature=i * 0.5)
        score = float_cvt(output)
        if score is None or not (0.0 <= score <= 1.0):
            log += f'Try {i}: output is {output}, failed or invalid.\n'
        else:
            log += 'Succeed'
            return dict(log=log, score=score)
    log += 'All retries failed.\n'
    return dict(log=log, score=0.0)


def Sanguo_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    cate2_list = []

    for _, item in data.iterrows():
        cate = item['category']
        cate2 = cate.replace(',', '_')
        if cate2 not in cate2_list:
            cate2_list.append(cate2)
        grade = float(item['score'])
        tot['Overall'] += 1
        score['Overall'] += grade
        tot[cate2] += 1
        score[cate2] += grade

    res = defaultdict(list)
    for k in cate2_list + ['Overall']:
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['acc'].append(score[k] / tot[k] * 100)
    return pd.DataFrame(res), pd.DataFrame(res)
