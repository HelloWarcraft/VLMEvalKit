from ...smp import *
import re
from collections import defaultdict
import pandas as pd


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


def parse_score(val):
    """从字符串中提取一个合法的 float 分数，并限定在 [0.0, 1.0] 之间"""
    if isinstance(val, float):
        score = val
    elif isinstance(val, int):
        score = float(val)
    elif isinstance(val, str):
        try:
            score = float(val)
        except ValueError:
            match = re.search(r'(\d(?:\.\d+)?)', val)
            if match:
                score = float(match.group())
            else:
                return 0.0
    else:
        return 0.0

    # 限定在合法范围内
    if 0.0 <= score <= 1.0:
        return score
    return 0.0


def Sanguo_auxeval(model, line):
    prompt = build_sanguo_prompt(line)
    log = ''
    for i in range(5):
        output = model.generate(prompt, temperature=i * 0.5)
        score = parse_score(output)
        if score == 0.0 and '0.0' not in str(output):
            log += f'Try {i}: output is {output}, failed or invalid.\n'
        else:
            log += 'Succeed'
            return dict(log=log, score=score)
    log += 'All retries failed.\n'
    return dict(log=log, score=0.0)


def Sanguo_acc(result_file):
    data = load(result_file)
    tot = defaultdict(int)
    score = defaultdict(float)
    cate2_list = []

    for idx, item in data.iterrows():
        cate = item['category']
        cate2 = cate.replace(',', '_')
        if cate2 not in cate2_list:
            cate2_list.append(cate2)

        raw_score = item['score']
        grade = parse_score(raw_score)

        # print(f"[DEBUG] row {idx} | category: {cate2} | raw_score: {raw_score} | parsed: {grade}")

        tot['总计得分'] += 1
        score['总计得分'] += grade
        tot[cate2] += 1
        score[cate2] += grade

    res = defaultdict(list)
    for k in cate2_list + ['总计得分']:
        res['Category'].append(k)
        res['total'].append(tot[k])
        res['sum_score'].append(round(score[k], 3))
        percentage = (score[k] / tot[k] * 100) if tot[k] > 0 else 0.0
        res['percentage_score'].append(f"{percentage:.2f}%")

    df = pd.DataFrame(res)
    return df, df