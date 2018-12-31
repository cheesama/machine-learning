import json
with open('KorQuAD_v1.0_train.json') as f:
    data = json.load(f)

qaData = data['data']


with open('KorQuAD_v1.0_train_preprocess.txt','w') as output:
    output.write('question\tanswer\tcontext\n')

    for eachData in qaData:
        for eachParagraph in eachData['paragraphs']:
            for eachQA in eachParagraph['qas']:

                if len(eachQA['question']) < 5:
                    continue

                if len(eachQA['answers'][0]['text']) < 5:
                    continue

                output.write(eachQA['question'].replace('\n', ' ').replace('\t',' '))
                output.write('\t')
                output.write(eachQA['answers'][0]['text'].replace('\n', ' ').replace('\t',' '))
                output.write('\t')
                output.write(eachParagraph['context'].replace('\n', ' ').replace('\t',' '))
                output.write('\n')
