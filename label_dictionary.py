import json
import os
import pandas as pd
from pathlib import Path
import math
from model import draw_box,load_image,face_location_encoding,knn_modelling,face_prediction,add_vector_location_img
from utils import special_layout,dict_5row_layout,col_layout,number_to_0000,write_table,create_annotated_dir,change_image_name
import matplotlib.pyplot as plt
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import Cursor, Button
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class Label_Dictionary():
    def __init__(self,classname,save_individual_annotated=False):
        global json_path
        global df_path
        global img_path 
        img_dir_path = f'./data/{classname}/image/individual'
        json_path = f'./data/{classname}/{classname}.json'
        df_path = f'./data/{classname}/{classname}.csv'
        self.classname = classname

        # read dictionary if exist
        if Path(json_path).is_file():
            with open(json_path,'r') as doc:
                label_dict = json.load(doc)

            # reminder
            print(special_layout(f'Loaded label dictionary: {json_path}\n***You can type {self.classname}.dict_ to access dictionary'))
            print(dict_5row_layout(label_dict,'name',blank = 15,each_row =5))
            self.dict_ = label_dict

        # make dictionary if not exist
        else:
            create_annotated_dir(self.classname)
            try:
                # 1. name
                print(special_layout(f'1. Insert name on {classname}.json'))
                dict_list = [ {'name': name, 'vector(individual)':[],'location(individual)':[],'img(individual)':[],'resize':[],
                                'vector(class)':[],'location(class)':[],'img(class)':[]} for name in pd.read_csv(df_path).columns[1:] ]
                label_dict = dict(zip(range(len(dict_list)),dict_list))
                print(dict_5row_layout(label_dict,'name',blank = 15,each_row =5))

                # 2. img
                print(special_layout(f'2. Insert img(individual) path on {classname}.json'))
                for img_path in sorted(Path(img_dir_path).glob("*.jpg")):
                    label_num = int(img_path.name[:4])
                    label_dict[label_num]['img(individual)'].append(str(img_path))
                    print(f'Image ({img_path.name}) append to {dict_list[label_num]["name"]}')
                


                # 3. location,vector
                print(special_layout(f'Insert face location and vector (individual) on {classname}.json\n***face location: (row1,col2,row2,col1)'))
               
                for each_dict in label_dict.values():
                    print(f'\nProcress image of {each_dict["name"]}:\n')
                    img_path_list = each_dict['img(individual)']
                    for img_path in img_path_list:
                        # load image
                        array = load_image(img_path)

                        # detect face and ecode
                        print(f'Detecting and Encoding face on image({img_path[-12:]})...')
                        location_list,vector_list = face_location_encoding(array)
                        print(f'{len(location_list)} Location append to {each_dict["name"]}')
                        print(f'{len(vector_list)} Vector append to {each_dict["name"]}')
                        vector_list = list(map(lambda array: list(array),vector_list))

                        if save_individual_annotated:
                            annotated_img = draw_box(array, location_list,show = False)
                            plt.imsave(img_path.replace('individual','annotated'),annotated_img)

                        # warning
                        if len(location_list)!=1:
                            words = f'Please check image({img_path[:-13]}) on {each_dict["name"]}!\n***amount of face on image (individual) is not one ({len(vector_list)})'
                            print(special_layout(special_layout(words)))

                        # location
                        each_dict['location(individual)']+=location_list
                        
                        # vector
                        each_dict['vector(individual)']+=vector_list

                    # add img path on extra face    
                    if len(location_list)>1:
                            for i in range(len(location_list)-1):
                                    each_dict['img(individual)'].append(img_path)
                                    i+=1


                # Final: output to dir
                with open(json_path,'w') as doc:
                    doc.write(json.dumps(label_dict))

                # reminder
                print(special_layout(f'Created label dictionary: {json_path}'))

                with open(json_path,'r') as doc:
                    label_dict = json.load(doc)

                self.dict_ = label_dict
                print('\nName: \n')
                print(dict_5row_layout(label_dict,'name',blank = 15,each_row =5))
                print('\nLocation: \n')
                print(dict_5row_layout(label_dict,'location(individual)',blank = 15,each_row =5,count_value = True))

            except:
                print(special_layout(f'Please creates attendence table: {df_path}\n\nPlease creates directory: {f"./{classname}"}'))

    
    
    def face_visualize(self, number = None, img = None, save = False, show = True):

        # create save dir
        if save:
            create_annotated_dir(self.classname)

        def print_face(i,img_path_list,location_list,save = False):
            array = load_image(img_path_list[i])
            annotated_array = draw_box(array, [location_list[i]],show = True)
            if save:
                plt.imsave(img_path.replace('individual','annotated'),annotated_array)
        '''
        number,img = None -> input(class or indivdiual)
        number != None -> all face of the individual
        '''

        
        # show
        # number != None
        if number != None:
            target_dict = self.dict_[str(number)]
            print(special_layout(f'Visualizing individual face on {target_dict["name"]}'))
            # all img
            for i in range(len(target_dict['img(individual)'])):
                    print_face(i,target_dict['img(individual)'],target_dict['location(individual)'],save)


        # img == None
        # specific face on one indiviual
        elif img != None:
            # indiviual
            if re.match(r'\d{4}_',img[0:4]):
                target_dict = self.dict_[str(int(img[0:4]))]
                i = list(map(lambda path: bool(re.search(img,path)),target_dict['img(individual)'])).index(True)
                print(special_layout(f'Visualizing individual face on {target_dict["name"]}'))
                print_face(i,target_dict['img(individual)'],target_dict['location(individual)'],save)

            if re.match(r'\d{4}-',img[0:4]):
                class_face_location_dict = {}
                for label,each_dict in self.dict_.items():
                    i = list(map(lambda path: bool(re.search(img,path)),each_dict['location(class)'])).index(True)
                    class_face_location_dict[label] = self.dict_[str(label)]['location(class)'][i]
                array = load_image("./data/FTDS5/image/class/"+img)
                draw_box(array, list(class_face_location_dict.values()), show = True 
                        ,label_test = list(class_face_location_dict.keys()), Dict = self.dict_)
        else:
            print(special_layout("Enter number or img"))
            return None
    
    def modelling(self,n_neighbors= 1):
        vector_train=[]
        label_train=[]
        vector_individual_train = []
        label_individual_train = []
        print(special_layout(f"Vector amount summary:"))
        print(col_layout('Label','Vector(individual) amount','Vector(class) amount','Total'))
        for label,each_dict in self.dict_.items():
            total = len(each_dict['vector(individual)'])+len(each_dict['vector(class)'])
            print(col_layout(str(label)+'.'+each_dict['name'],len(each_dict['vector(individual)']),len(each_dict['vector(class)']),total))
            vector_train = vector_train + each_dict['vector(individual)']+each_dict['vector(class)']
            vector_individual_train += each_dict['vector(individual)']
            label_train += [int(label) for i in range(total)]
            label_individual_train += [int(label)]
        knn_modelling(self.classname,vector_individual_train,label_individual_train,n_neighbors=1,only_individual = True)
        knn_modelling(self.classname,vector_train,label_train,n_neighbors=n_neighbors)

    def tick_attendence(self,img_name = None,save_annotated = True,add_vector = True,n_neighbors = 1):
        global Label_test
        global location_list
        global vector_list
        global mode
        def line_select_callback(click,release):
            global Label_test
            global location_list
            global target
            row1,col2,row2,col1 = int(click.ydata),int(release.xdata),int(release.ydata),int(click.xdata)
            Label_test.append(int(target))
            location_list.append((row1,col2,row2,col1))
            print(special_layout(f"Added {self.dict_[str(target)]['name']} to annotated image.\n\
Amount of targets: {len(location_list)}"))
            plt.close()

        def onclick(event):
            global Label_test
            global location_list
            global vector_list
            global mode

            col, row = event.xdata, event.ydata
            for i in range(len(location_list)):
                    row1,col2,row2,col1 = location_list[i] 
                    if row > row1 and row < row2 and col > col1 and col < col2:
                        if mode == '2':
                            try:
                                correction = input(special_layout(f"You select {self.dict_[str(Label_test[i])]['name']} ({Label_test[i]})\n***correction -> 1 No correction -> 0")) 
                                if int(correction):

                                        correct_label = input(special_layout(f"Who is this? :\n\n\
{dict_5row_layout(self.dict_,'name',blank = 15,each_row =5,count_value = False)}\n***Please type number"))
                                        print(special_layout(f"{self.dict_[str(Label_test[i])]['name']} -> {self.dict_[str(correct_label)]['name']}"))
                                        Label_test[i] = correct_label
                            except:
                                pass
                            break
                        elif mode == '3':
                            try:
                                delete = input(special_layout(f"You confirm to delete {self.dict_[str(Label_test[i])]['name']} ({Label_test[i]}) on [{row1}:{row2},{col1}:{col2}]?\n***yes -> 1 No no -> 0"))
                            except:
                                pass
                            break
            try:
                if int(delete):
                    Label_test.pop(i)
                    location_list.pop(i)
                    vector_list.pop(i)
            except:
                pass
            plt.close()

        def toggle_selector(event):
                toggle_selector.RS.set_active(True)

        def object_mode_change(event):
            global target
            global mode
            if event.key == '1':
                mode = '1'
                print(special_layout(f" Add label on annotated image."))
                try:
                    target = input(special_layout(f"Select target label:\n\n\
{dict_5row_layout(self.dict_,'name',blank = 15,each_row =5,count_value = False)}"))
                    print(f"You will annotate {self.dict_[str(target)]['name']} ({target})")
                    plt.close()
                except:
                    pass
            elif event.key == '2':
                mode = '2'
                print(special_layout(f" Change label on annotated image."))
                plt.close()

            elif event.key == '3':
                mode = '3'
                print(special_layout(f" Delete label on annotated image."))
                plt.close()

            elif event.key == 'q':
                mode = 'q'
                print(special_layout(f"Finish correction..."))
                plt.close()
            
            else:
                print(special_layout(f"Please press the followings key:\n\nAdd annotation -> 1\nClick show label and change label -> 2\nDelete annotation -> 3\n\
Exit correction -> q"))

        # change name
        if img_name == None:
            try:
                img_name = change_image_name(self.classname)
            except:
                img_name = [path for path in sorted(Path(f"./data/{self.classname}/image/class").glob("*.jpg"))][-1].name

        img_path = f'./data/{self.classname}/image/class/{img_name}'

        print(special_layout(f"Detect and encode faces on image({img_name})"))
        array = load_image(img_path)
        location_list, vector_list = face_location_encoding(array)
        Label_test = list(face_prediction(self.classname, vector_list))
        annotated = draw_box(load_image(img_path), location_list,False,Label_test ,self.dict_)
        count = 0
        while True:
            print(special_layout(f"Show you the annotated image...\nEnter q"))
            fig, ax = plt.subplots(1)
            plt.imshow(annotated)
            plt.show()
            if count%2==0:
                indivdual = input(special_layout(f"Try individual model?"))
                if int(indivdual):
                    Label_test = list(face_prediction(self.classname, vector_list,only_individual=True))
                    annotated = draw_box(load_image(img_path), location_list,False,Label_test ,self.dict_)
                else:
                    break
            else:
                back = input(special_layout(f"Try the previous model?"))
                if int(back):
                    Label_test = list(face_prediction(self.classname, vector_list))
                    annotated = draw_box(load_image(img_path), location_list,False,Label_test ,self.dict_)
                    plt.close()
                else:
                    print(123)
                    break
            count += 1
        print(special_layout(f"Show you the annotated image...\nPress H to watch instruction:)"))
        mode = '2'
        while True:
            annotated = draw_box(load_image(img_path), location_list,False,Label_test ,self.dict_)
            fig, ax = plt.subplots(1)
            plt.imshow(annotated)

            if mode == '1':
                toggle_selector.RS = RectangleSelector(
                    ax,line_select_callback,
                    drawtype='box',useblit=True,
                    button=[1],minspanx=5,minspany=5,
                    spancoords='pixels',interactive=True
                    )

                plt.connect('key_press_event', toggle_selector)
                plt.connect('key_press_event',object_mode_change)

            elif mode == '2' or mode == '3':
                Cursor(ax,
                horizOn=False, # Controls the visibility of the horizontal line
                vertOn=False, # Controls the visibility of the vertical line
                )
                fig.canvas.mpl_connect('button_press_event', onclick)
                plt.connect('key_press_event',object_mode_change)
            
            plt.show()
            if mode == 'q':
                break
                
        if save_annotated:
            create_annotated_dir(self.classname)
            plt.imsave(img_path.replace('class','annotated'),annotated)
        # write table
        write_table(self.dict_,self.classname,Label_test,img_name)

        # add vector # Modelling
        if add_vector:
            vector_correct = input(special_layout(f"Add all face into our knn model?\n***yes -> 1 no -> 0"))
            if int(vector_correct)-1:
                mode = '2'
                while True:
                    annotated = draw_box(load_image(img_path), location_list,False,Label_test ,self.dict_)
                    fig, ax = plt.subplots(1)
                    plt.imshow(annotated)

                    if mode == '1':
                        toggle_selector.RS = RectangleSelector(
                            ax,line_select_callback,
                            drawtype='box',useblit=True,
                            button=[1],minspanx=5,minspany=5,
                            spancoords='pixels',interactive=True
                            )

                        plt.connect('key_press_event', toggle_selector)
                        plt.connect('key_press_event',object_mode_change)

                    elif mode == '2' or mode == '3':
                        Cursor(ax,
                        horizOn=False, # Controls the visibility of the horizontal line
                        vertOn=False, # Controls the visibility of the vertical line
                        )
                        fig.canvas.mpl_connect('button_press_event', onclick)
                        plt.connect('key_press_event',object_mode_change)
                    
                    plt.show()
                    if mode == 'q':
                        break

            add_vector_location_img(self.dict_,self.classname,vector_list,Label_test,location_list,img_name)
            vector_train=[]
            label_train=[]
            print(special_layout(f"Vector amount summary:"))
            print(col_layout('Label','Vector(individual) amount','Vector(class) amount','Total'))
            for label,each_dict in self.dict_.items():
                total = len(each_dict['vector(individual)'])+len(each_dict['vector(class)'])
                print(col_layout(str(label)+'.'+each_dict['name'],len(each_dict['vector(individual)']),len(each_dict['vector(class)']),total))
                vector_train = vector_train + each_dict['vector(individual)']+each_dict['vector(class)']
                label_train += [int(label) for i in range(total)]

            knn_modelling(self.classname,vector_train,label_train,n_neighbors =n_neighbors)

        # Final: output to dir
        with open(json_path,'w') as doc:
            doc.write(json.dumps(self.dict_))

        # reminder
        print(special_layout(f'Renew label dictionary: {json_path}'))
        

    def print_all_info(self,key,blank = 15,each_row =5, count_value = False):
       if count_value:
           word = ' count'
       else:
            word = ''
       print( f'Printing {key + word} of {self.classname}\n' )
       print(dict_5row_layout(self.dict_,key,blank = 15,each_row =5,count_value = count_value))


    def update(self,number,mode = 'individual'):
        if mode == 'individual':
            print(special_layout(f"Updating {self.dict_[str(number)]['name']}  individual vector, location and img..."))
            self.dict_[str(number)]['img(individual)'] = []
            self.dict_[str(number)]['vector(individual)'] = []
            self.dict_[str(number)]['location(individual)'] = []
            for img_path in Path('./data/FTDS5/individual').glob(f'{number_to_0000(number)}*.jpg'):
                self.dict_[str(number)]['img(individual)'].append(str(img_path))
            for img_path in self.dict_[str(number)]['img(individual)']:
                        # load image
                        array = load_image(img_path)
                        # detect face and ecode
                        print(f'Detecting and Encoding face on image({img_path[:-13]})...')
                        location_list,vector_list = face_location_encoding(array)
                        print(f'{len(location_list)} Location append to {self.dict_[str(number)]["name"]}')
                        print(f'{len(vector_list)} Vector append to {self.dict_[str(number)]["name"]}')
                        vector_list = list(map(lambda array: list(array),vector_list))

                        # warning
                        if len(location_list)!=1:
                            words = f'Please check image({img_path[:-13]}) on {self.dict_[str(number)]["name"]}!\n***amount of face on image (individual) is not one ({len(vector_list)})'
                            print(special_layout(special_layout(words)))

                        # location
                        self.dict_[str(number)]['location(individual)']+=location_list
                        
                        # vector
                        self.dict_[str(number)]['vector(individual)']+=vector_list

                        # add img path on extra face    
                        if len(location_list)>1:
                                for i in range(len(location_list)-1):
                                        self.dict_[str(number)]['img(individual)'].append(img_path)
                                        i+=1
            print(special_layout(f'Updated {self.classname} vector, location and img (individual)...'))



    
if __name__ == '__main__':
    Avengers = Label_Dictionary('Avengers')
    Avengers.modelling(n_neighbors=1)
    for path in sorted(Path("/Users/15077693d/Desktop/FTDS/GitHub/Attendancv/data/Avengers/image/class").glob('*.jpg')):
        Avengers.tick_attendence(img_name=path.name)
   