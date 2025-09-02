import json

raw_data = json.load(open('question_decompositions_100.json'))

def check(question):
    if '#1' in question or '#2' in question or '#3' in question or '#4' in question:
        return True
tree = {}
for father in raw_data:
    if check(father):
        continue
    qds = raw_data[father]
    if qds is None:
        continue
    tree[father] = {}
    valid_questions = False
    for question in qds:
        if check(question):
            continue
        if any([x == question for x in qds[question][0]]):
            tree[father][question] = [[], None]
        else:
            tree[father][question] = qds[question]
        valid_questions = True
    # 如果没有有效的子问题，保留根节点
    if not valid_questions:
        tree[father] = {father: qds.get(father, [[], None])}

question_decompositions = {}
for father in tree:
    qds = tree[father]
    for q in qds:
        if q not in question_decompositions:
            question_decompositions[q] = qds[q]
        else:
            if question_decompositions[q] != qds[q]:
                print(question_decompositions[q])
                print(qds[q])
            else:
                print('haha')

json.dump(question_decompositions, open('tree_100.json', 'w'), indent = 2)

print(len(tree))