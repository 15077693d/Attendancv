import pandas as pd
import json
from datetime import datetime
import re
from pathlib import Path
import os

def special_layout(words):
        line = '---------------------------------------------------------------------------------------'
        return '\n' + line + '\n\n' + words + '\n\n' + line + '\n'

def dict_5row_layout(Dict,key,blank = 15,each_row =5,count_value = False):
    words = ''
    count = 1
    for key_,value in Dict.items():
        item = value[key]
        if count_value:
            item = len(value[key])
        word = str(key_) + '.' + str(item)
        space_amount = blank - len(word)
        word += ' '*space_amount
        words += word
        if count%each_row == 0:
            words+='\n'
        count+=1
    return words+'\n'

def col_layout(*argv,blank =30):
            output = ""
            for word in argv:
                space = blank-len(str(word))
                output+=(str(word)+space*" ")
            return output

def number_to_0000(number):
            if int(number)<10:
                    return '000'+str(number)
            if int(number)<100:
                    return '00'+str(number)
            if int(number)<1000:
                    return '0'+str(number)

def write_table(Dict,classname,Label_test,img_name):
    df = pd.read_csv(f'./data/{classname}/{classname}.csv')
    time = img_name.replace('.jpg','')
    same_index = []
    for i in range(len(df.time)):
        if df.time[i] == time:
            same_index.append(i)
    if len(same_index)>0:
            delete = input(special_layout(f"There have {len(same_index)} same record at {img_name.replace('.jpg','')}, remove previous one?\n***yes -> 1 no -> 0"))
            if delete:
                df = df.drop(index = same_index)
    name_list = [Dict[str(label)]['name'] for label in Label_test]
    new_row = [time] + [1 if (name in name_list) else 0 for name in df.columns[1:]]
    new_df = pd.DataFrame([new_row],columns = df.columns)
    df = pd.concat([df,new_df],axis=0)
    print('\n',df)
    output = input(special_layout("Output dataframe?\n***yes -> 1 no -> 0"))
    if int(output):
        df.to_csv(f'./data/{classname}/{classname}.csv',index =False)

def create_annotated_dir(classname):
            save_path = f'./data/{classname}/image/annotated'
            try:
                os.mkdir(save_path)
                print(f'Directory ceated {save_path}')
            except:
                print(f'Directory exist {save_path}')

def change_image_name(classname):
    now = datetime.now()
    image_name = now.strftime("%Y-%m-%d %H:%M")+'.jpg'
    path_list = sorted(Path(f'./data/{classname}/image/class').glob("*.jpg"))+sorted(Path(f'./data/{classname}/image/class').glob("*.JPG"))
    for path in path_list:
        if bool(re.search(r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}.jpg",path.name))-1:
            print(special_layout(f"Changing image name: {path.name} -> {image_name}"))
            os.rename(str(path),str(path).replace(path.name,image_name))
            correct = 1
            break
    if correct:
        return image_name
        
if __name__ == "__main__":
    classname = 'FTDS5'
    with open('./data/FTDS5/FTDS5.json') as doc:
        Dict = json.load(doc)
    Label_test = [1,3,4,5,10,11,12,13,14,15]
    img_name = '19-10-06 09:24.jpg'
    write_table(Dict,classname,Label_test,img_name)