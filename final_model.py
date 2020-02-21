from __future__ import unicode_literals, print_function


import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import logging
import json
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score


required_details_model1 = ['Name','Email Address','Degree','University','College Name','Graduation Year','Certifications',
            'Rewards and Achievements','Designation','Companies worked at','Years of Experience',
            'Skills','work experience','projects']
required_details_model2 = ["Rewards and Achievements","Skills","Years of Experience","projects","Links"]

def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        testing_data = []
        lines=[]
        with open(dataturks_JSON_FilePath, 'r' , encoding='utf8') as f:
            lines = f.readlines()

        for i,line in enumerate(lines):
            data = json.loads(line)
            text = data['content']
            entities = []
            try:
                for annotation in data['annotation']:
                    #only a single point in text annotation.
                    point = annotation['points'][0]
                    labels = annotation['label']
                    # handle both list of labels or a single label.
                    if not isinstance(labels, list):
                        labels = [labels]
    
                    for label in labels:
                        if label in required_details_model2:
                        #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                            entities.append((point['start'], point['end'] + 1 ,label))
                        else:
                            continue
                        #converting_entities(label)
            except:
                continue
            if i%10 == 0:
                testing_data.append((text, {"entities" : entities}))
            else:
                training_data.append((text, {"entities" : entities}))

        return training_data,testing_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None

TRAIN_DATA = convert_dataturks_to_spacy("F:\\ATS\\Final_ATS\\resume_data\\Entity Recognition in Resumes.json")
with open("F:\\ATS\\Final_ATS\\training_data\\data_collected1.json", "w") as write_file:
    json.dump(TRAIN_DATA, write_file)
def model_testing(nlp2,examples):
#    examples = convert_dataturks_to_spacy("F:\\ATS\\Final_ATS\\resume_data\\Entity Recognition in Resumes.json")
    tp=0
    tr=0
    tf=0

    ta=0
    c=0        
    for text,annot in examples:
        if c<50:

            f=open("resume"+str(c)+".txt","w",encoding='utf-8')
            doc_to_test=nlp2(text)
            d={}
            for ent in doc_to_test.ents:
                d[ent.label_]=[]
            for ent in doc_to_test.ents:
                d[ent.label_].append(ent.text)
    
            for i in set(d.keys()):
    
                f.write("\n\n")
                f.write(i +":"+"\n")
                for j in set(d[i]):
                    f.write(j.replace('\n','')+"\n")
            d={}
            for ent in doc_to_test.ents:
                d[ent.label_]=[0,0,0,0,0,0]
            for ent in doc_to_test.ents:
                try:
                    doc_gold_text= nlp2.make_doc(text)
                    gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
                    y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
                    y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
                    if(d[ent.label_][0]==0):
                        #f.write("For Entity "+ent.label_+"\n")   
                        #f.write(classification_report(y_true, y_pred)+"\n")
                        (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
                        a=accuracy_score(y_true,y_pred)
                        d[ent.label_][0]=1
                        d[ent.label_][1]+=p
                        d[ent.label_][2]+=r
                        d[ent.label_][3]+=f
                        d[ent.label_][4]+=a
                        d[ent.label_][5]+=1
                except:
                    continue
            c+=1
    for i in d:
        print("\n For Entity "+i+"\n")
        print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
        print("Precision : "+str(d[i][1]/d[i][5]))
        print("Recall : "+str(d[i][2]/d[i][5]))
        print("F-score : "+str(d[i][3]/d[i][5]))


def main(model=None, output_dir=None, n_iter=10):
    """Load the model, set up the pipeline and train the entity recognizer."""
    TRAIN_DATA ,TESTING_DATA= convert_dataturks_to_spacy("F:\\ATS\\Final_ATS\\resume_data\\Entity Recognition in Resumes.json")
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
            optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                try:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses,
                    )
                except:
                    continue
            print("Losses", losses)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)
    model_testing(nlp2,TESTING_DATA)

import extract_text
import os
from spacy.matcher import Matcher
import utils
def test_resume(folderpath,model):
    for file in os.listdir(folderpath):
        details = {}
        basic = ['Name','Email Address','Phone No']
        obtained = []
        raw_text = extract_text.get_extracted_data(os.path.join(folderpath,file))
        #doc = nlp2(raw_text)
        nlp = spacy.load('en_core_web_sm')
        matcher = Matcher(nlp.vocab)
        nlp_text = nlp(raw_text)
        details['Email Address'] = utils.extract_email(raw_text)
        details['Phone No'] = utils.extract_mobile_number(raw_text)
        details['Name'] = utils.extract_name(nlp_text, matcher)
        f=open("resume_"+file+".txt","w",encoding='utf-8')
        doc_to_test=model(raw_text)
        d={}
        for ent in doc_to_test.ents:
            d[ent.label_]=[]
            obtained.append(ent.label_)
        remaining = list(set(basic) - set(obtained))
        print(remaining)
        for ent in doc_to_test.ents:
            d[ent.label_].append(ent.text)
        
        for i in remaining:

            f.write("\n\n")
            f.write(i +":"+"\n")
            try:
                f.write(details[i])
            except:
                continue
            
                
                
                
        for i in set(d.keys()):

            f.write("\n\n")
            f.write(i +":"+"\n")
            for j in set(d[i]):
                f.write(j.replace('\n','')+"\n")
        f.write("############### Raw test ###################"+"\n\n")
        f.write(raw_text)

if __name__ == "__main__":
    nlp2 = spacy.load('F:\\ATS\\Final_ATS\\final_model')
    test_resume('F:\\ATS\\resumes',nlp2)
    main(None,"F:\\ATS\\Final_ATS\\model4",50)
    nlp2 = spacy.load('F:\\ATS\\Final_ATS\\model3')
    nlp2.meta
    
    doc = nlp2("   Harini Komaravelli\nTest Analyst at Oracle")
    print ("Entities= " + str(["" + str(ent.text) + "_" + str(ent.label_) for ent in doc.ents]))