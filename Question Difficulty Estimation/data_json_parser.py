#-*- coding:utf-8 -*-
import json
import csv
import os

def json_load(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)

def writer_tsv(output_file, data_list):
    with open(output_file+'.tsv', "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='\t')
        for data in data_list:
            writer.writerow(data)

if __name__=="__main__":

    prefix_data_dir = "/home/shyun/workspace/data/drama/AnotherMissOh"
    data_name_list = ["AnotherMissOhQA_train_set.json", "AnotherMissOhQA_val_set.json"]
    script_file = "AnotherMissOh_script.json"
    description_file = "AnotherMissOh_description.json"
    
    description_data = json_load(description_file)
    script_data = json_load(script_file)

    for data_name in data_name_list:
        cur_data = json_load(data_name)
        inform = []
        for data in cur_data:
            #extract basic information data
            qid = data["qid"]
            vid = data["vid"]
            q_level_mem = data['q_level_mem']
            q_level_logic = data['q_level_logic']
            que = data['que']
            answers = data['answers']
            correct_answer = answers[data['correct_idx']]
            shot_contained_number = data["shot_contained"]

            #extract utterance
            utterance = '[UTR]'
            
            if vid[-4:] == "0000":
                for shot_num in shot_contained_number:
                    length = len(str(shot_num))
                    if length == 1:
                        shot_id = vid[:-4]+'000'+str(shot_num)
                    elif length == 2:
                        shot_id = vid[:-4]+'00'+str(shot_num)
                    elif length == 3:
                        shot_id = vid[:-4]+'0'+str(shot_num)
                    elif length == 4:
                        shot_id = vid[:-4]+str(shot_num)
                    else:
                        shot_id = None
                    try:
                        container = script_data[shot_id]["contained_subs"]
                        for contain in container:
                           utterance += contain["utter"].replace('\n','')
                    except KeyError:
                        pass
            else:
                try:
                    container = script_data[vid]["contained_subs"]
                    for contain in container:
                        utterance += contain["utter"].replace('\n','')
                except KeyError:
                    pass

            #extract description
            description = ''
           
            #if vid[-4:] == "0000":
            #    for shot_num in shot_contained_number:
            #        length = len(str(shot_num))
            #        if length == 1:
            #            shot_id = vid[:-4]+'000'+str(shot_num)
            #        elif length == 2:
            #            shot_id = vid[:-4]+'00'+str(shot_num)
            #        elif length == 3:
            #            shot_id = vid[:-4]+'0'+str(shot_num)
            #        elif length == 4:
            #            shot_id = vid[:-4]+str(shot_num)
            #        else:
            #            shot_id = None
            #        
            #        for text in description_data[shot_id]:
            #            description += text.replace('\n','') 
            #else:
            for text in description_data[vid]:
                description += text.replace('\n','')

            inform.append([qid, que, description, answers, correct_answer, utterance, q_level_mem, q_level_logic])
        
        writer_tsv(data_name.split('_')[1], inform)


