#coding:utf-8

import os
import glob
import numpy as np
from PIL import Image
import shutil

if __name__ == '__main__':
    pic_path = "pic"
    if os.path.exists(pic_path):
        shutil.rmtree(pic_path)
    else:
        os.mkdir(pic_path)
        
    if os.path.isdir(pic_path):
        pass
    else:
        os.mkdir(pic_path)
        
    json_path = "json"
    if os.path.exists(json_path):
        shutil.rmtree(json_path)
    else:
        os.mkdir(json_path)

    if os.path.isdir(json_path):
        pass
    else:
        os.mkdir(json_path)
        
    mask_path = "cv2_mask"
    if os.path.exists(mask_path):
        shutil.rmtree(mask_path)
    else:
        os.mkdir(mask_path)

    if os.path.isdir(mask_path):
        pass
    else:
        os.mkdir(mask_path)
        
    labelme_json = "labelme_json"
    if os.path.exists(labelme_json):
        shutil.rmtree(labelme_json)
    else:
        os.mkdir(labelme_json)

    if os.path.isdir(labelme_json):
        pass
    else:
        os.mkdir(labelme_json)
        
        
    sku_path = os.getcwd()
    # 处理labelme生成的json文件
    json_lsts = glob.glob(sku_path + "/*.json")
    for i in json_lsts:
        str_commod = "labelme_json_to_dataset" + " " + i
        os.system(str_commod)
        
        shutil.move(i , os.path.join(json_path , os.path.basename(i)))
        shutil.move(i[:-5] + "_json" , labelme_json)
        
        
        shutil.move(i[:-4] + "png" , os.path.join(pic_path , os.path.basename(i)[:-4] + "png"))
    
    # 保存8位的Mask文件
    json_dirs = os.listdir(labelme_json)
    for i in json_dirs:
        tmp = os.path.join(labelme_json , i , "label.png")
        if os.path.exists(tmp):
            img = Image.open(tmp)
            img = Image.fromarray(np.uint8(img))
            img_name = i.split('_')[0]
            img.save(os.path.join(mask_path , img_name + ".png"))
            

    print('finished!')
