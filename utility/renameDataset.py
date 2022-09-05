import os
import shutil

#   path = "dataset/Polipi/masks/"
#   path = "dataset/Polipi/dataset/train/0_normal/"

# dataset_path
# masks_path
# sets

def process_rename(dataset_path, set = None, label = None):
    
    i = "0000"
    prefix = None
    mask_from = None
    mask_to = dataset_path + 'mask/'
    dataset_path = dataset_path + 'image/'
    dataset_image = dataset_path
    
    if set == None:
        mask_from = dataset_path + 'mask/'
        prefix = "0000000"
    else:
        if label == None:
            mask_from = dataset_path + set + '/' + 'mask/'
            prefix = set + "_" + "0000000"
            dataset_image = dataset_image + set + '/'
        else:
            mask_from = dataset_path + set + '/' + 'mask/'
            prefix = label
            while len(prefix) < 7:
                prefix = '0' + prefix
            prefix = set + "_" + prefix
            dataset_image = dataset_image + set + '/'
            
    dataset_image = dataset_image + prefix + '/'
                 
    if label == None:
        if not os.path.exists(dataset_image):
            os.makedirs(dataset_image)
        for file in os.listdir(dataset_path + set):
            if file != 'mask' and file != prefix:
                shutil.move(dataset_path + set + '/' + file, dataset_image)
    else:
        os.rename(dataset_path + set + '/' + label, dataset_image)
    
    if not os.path.exists(mask_to):
        os.makedirs(mask_to)  
        
    for image in os.listdir(dataset_image):
        if image.endswith(".png"):
            #print("move image: ", dataset_image + image, " to " , dataset_image + prefix)
            #print("rename image: ", dataset_image + image, " to " ,dataset_image + prefix + '_' + i + ".png")
            #print("move mask: ", mask_from + image," to ", mask_to)
            #print("rename mask: ", mask_to + image," to ", mask_to + prefix + '_' + i + "_imag_res_fill.png")

            #move image
            if label == '':
                shutil.move(dataset_image + image, dataset_image + prefix)
            #rename image
            os.rename(dataset_image + image, dataset_image + prefix + '_' + i + ".png")
            #move mask
            shutil.move(mask_from + image, mask_to)
            #rename mask
            os.rename(mask_to + image, mask_to + prefix + '_' + i + "_imag_res_fill.png")
            i = str(int(i) + 1)
            while len(i) < 4:
                i = "0" + i
        if i == "10000":
            i = "0000"
            prefix = str(int(prefix[-7:]) + 1)
            while len(prefix) < 7:
                prefix = "0" + prefix
            prefix = set + prefix
            if not os.path.exists(dataset_image + prefix):
                os.mkdir(dataset_image + prefix)

        
def rename_dataset(dataset_path):
        
    found_sets = []
    found_mask = False

    for item in os.listdir(dataset_path):
        if os.path.isdir(dataset_path + item):
            if item != 'mask' and item != 'image':
                found_sets.append(item)
            else:
                found_mask = True
    
    if found_sets == []:
        print("No directory found, the rename will be done for images found in dataset_path")
    else:
        print("At least one directory is found, the rename will be done for those directories")
    
    if not os.path.exists(dataset_path + 'image'):
            os.mkdir(dataset_path + 'image')
    
    print("Found sets: " + str(found_sets))
    if found_sets == []:  
        if not found_mask: 
            print("No mask directory found, you must give a mask directory inside dataset_path called 'mask'")
            return
        os.mkdir(dataset_path  + 'image/' + "train/")
        for image in os.listdir(dataset_path):
            if image != 'image':
                shutil.move(dataset_path + image, dataset_path + 'image/' + "train/")
        process_rename(dataset_path, 'train')
        os.rmdir(dataset_path + 'image/' + 'train' + '/mask')
    else:
        for set in found_sets:
            found_mask = False
            found_label = []
            for item in os.listdir(dataset_path + set):
                if os.path.isdir(dataset_path + set + '/' + item):
                    if item != 'mask':
                        found_label.append(item)
                    else:
                        found_mask = True
                        
            shutil.move(dataset_path + set, dataset_path + 'image')
            if found_label == []:
                print("No label directory found for set: ",set)
                if not found_mask:
                    print("No mask directory found for set: ",set, " , you must give a mask directory called 'mask'")
                    continue
                print(set, "found mask dir")
                process_rename(dataset_path, set)
            else:
                print("Found labels: " + str(found_label))
                if not os.path.exists(dataset_path + 'image/' + set + '/' + 'mask'):
                    print("No mask directory found for set: ",set, " , you must give a mask directory called 'mask'")
                    continue
                for label in found_label:
                    process_rename(dataset_path, set, label)
            os.rmdir(dataset_path + 'image/' + set + '/mask')
    print("Done")
 
def rename_from_morph(dataset_path):
    for set in os.listdir(dataset_path):
        print("Processing set: ", set)
        for label in os.listdir(dataset_path + set):
            name = label[-7:]
            for i in name:
                if i != '0':
                    name = name[name.index(i):]
            os.rename(dataset_path + set + '/' + label, dataset_path + set + '/' + name)
    print("Done")
    