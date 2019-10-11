import cv2
import face_recognition
from mtcnn.mtcnn import MTCNN
from pathlib import Path
import matplotlib.pyplot as plt
import time
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
import pickle
from utils import special_layout,col_layout
import numpy as np
import os

def load_image(url,resize = False,size = 0.5):
        array = io.imread(url)
        print(f'Image path: {url}')
        print(f'Image shape : {array.shape}')
        if resize and array.shape[0] > 2000 and array.shape[1]*size > 2000:
                y = int(array.shape[0]*size)
                x =  int(array.shape[1]*size)
                array = cv2.resize(array,(x,y))
                print(f'Image shape after resize: {array.shape}')
        # plt.imshow(array)
        # plt.show(block=False)
        # input('continue?')
        # plt.close('all')
        return array

def face_location_encoding(array):
        def translate_box(box):
                row1 = box[1]
                row2 = row1 + box[3]
                col1 = box[0]
                col2 = col1 + box[2]
                return (row1,col2,row2,col1)
        start = time.time()
        # top, right, bottom, left
        FaceModel = MTCNN(steps_threshold=[0.5, 0.6, 0.9])
        output_list = FaceModel.detect_faces(array)
        location_list = list(map(lambda output_dict: translate_box(output_dict['box']),output_list))
        # print('\nThe Location: ',location_list,'\n')
        # print('\n',type(location_list),'\n')
        vector_list =  face_recognition.face_encodings(array,known_face_locations=location_list,num_jitters=10)
        # print('\nThe vector: ',vector_list[0],'\n')
        # print('\nThe vector amount: ',len(vector_list),'\n')
        # print(type(vector_list))
        print(f'Compute executing time: {str(time.time()-start)[:5]}')
        return (location_list,vector_list)

def shape_parameter_size(area):
        '''
        return (thickness,text_size,text_space,circle_radius)
        '''
        if area <= 100000:
                # line & font
                text_size = 0.5
                space = 2
                thickness = 1
                # point
                circle_radius = 2
        elif area<=600000:
                # line & font
                text_size = 0.6
                space = 4
                thickness = 2
                # point
                circle_radius = 2
        elif area < 1000000:
                # line & font
                text_size = 0.6
                space = 9
                thickness = 3
                # point
                circle_radius = 4
        elif area < 5000000:
                # line & font
                text_size = 1.5
                space = 10
                thickness = 2
                # point
                circle_radius = 6
               
        else:
                # line & font
                text_size = 2
                space = 9
                thickness = 3
                # point
                circle_radius = 10
        return (thickness,text_size,space,circle_radius)



def draw_box(array, location_list, show = True ,label_test = None, Dict = None,text_size = 0.6,thickness=2):
        area = array.shape[0]*array.shape[0]
        thickness,text_size,text_space,circle_radiu =shape_parameter_size(area)
        for i in range(len(location_list)):
                row1,col2,row2,col1 = location_list[i]
                cv2.rectangle(array,(col1,row1),(col2,row2),(0,255,0),thickness,cv2.LINE_AA)
                try:
                  cv2.putText(array,Dict[str(label_test[i])]['name'],(col1,row1),cv2.FONT_HERSHEY_SIMPLEX,text_size, (0, 255, 0), thickness, cv2.LINE_AA)
                except:
                        pass
        if show:
                plt.imshow(array)
                plt.show(block=False)
                input('\nStop to show image?\n')
                plt.close()
        return array

def knn_modelling(classname,vector_train,label_train,n_neighbors = 1,only_individual = False):
        if only_individual:
                name = f"{classname}_individual_knn"
                n_neighbors = 1
                if Path(f'./data/{classname}/{name}').is_file():
                        special_layout(f'KNN individual model (n_neighbors=1) exist ./data/{classname}/{name}\n')
                        return None
        if only_individual==False:
                name = f"{classname}_knn_{n_neighbors}"
        try:
                model_path = list(Path(f'./data/{classname}').glob(f'{classname}_knn_*'))[0]
                os.remove(str(model_path))
        except:
                pass
        print(special_layout(f'Created knn model for {classname} \n'))
        knn =  KNeighborsClassifier(n_neighbors=n_neighbors).fit(vector_train,label_train)
        knnPickle = open(f'./data/{classname}/{name}','wb')
        pickle.dump(knn, knnPickle)
        print(special_layout(f'KNN model (n_neighbors={n_neighbors}) outputed ./data/{classname}/{name}\n'))
        
def face_prediction(classname,vector_test,only_individual = False):
     print(special_layout(f"Loading {classname}_knn model..."))
     if only_individual:
        knn = pickle.load(open(f'./data/{classname}/{classname}_individual_knn','rb'))
     else:
        model_path = list(Path(f'./data/{classname}').glob(f'{classname}_knn_*'))[0]
        knn = pickle.load(open(model_path,'rb'))
     label_test = knn.predict(vector_test)
     print(f"Finished prediction...")
     return label_test

def add_vector_location_img(Dict,classname,vector_list,Label_list,location_list,img_name):
        img = f'./data/{classname}/image/class/{img_name}'
        print(f"\nIn image ({img_name}):\n")

        for i in range(len(vector_list)):
                Dict[str(Label_list[i])]['img(class)'].append(img)
                Dict[str(Label_list[i])]['location(class)'].append(list(location_list[i]))
                Dict[str(Label_list[i])]['vector(class)'].append(list(vector_list[i]))
                print(f"Added image,vector and location on {Dict[str(Label_list[i])]['name']} dictionary...")

if __name__ == '__main__':
        ts = 1
        th = 10

       
        for url in sorted(Path('/Users/15077693d/Desktop/FTDS/GitHub/Attendancv/data/Avengers/image/class').glob('*.jpg')):
                print(str(url))
                array = load_image(str(url))
                location_list,vector_list = face_location_encoding(array)
                while True:
                        Dict = {"1" :{'name': 'Oscar'}}
                        annotated_img = draw_box(load_image(str(url)), location_list, show = False ,
                                        label_test = ["1" for i in range(len(vector_list))], Dict = Dict,
                                        text_size = float(ts),thickness=int(th))
                        plt.imsave(str(url).replace('class','123'),annotated_img) 
                        ans = input(f"Currect size({annotated_img.shape[0]*annotated_img.shape[1]}): text_size({ts}),thickness({th})\n\n")
                        if ans=="q":
                                break
                        ts,th = ans.split(',')