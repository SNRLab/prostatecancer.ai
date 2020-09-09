import os
import sys
sys.path.append("../")
from glob import glob
from utils.helpers import *
#import jsonSa
import SimpleITK as sitk
#import settings as S
from keras.models import model_from_json
import numpy as np
import nrrd
class Deploy:
            
    def __init__(self):
        #self.current_dir = os.path.dirname(__file__) 
        self.current_dir = "C:/w/s/prostatecancer.ai-master/models/Densenet_T2_ABK_auc_079_nozone"
        self.datagen_dict = read_json(self.current_dir + "/configs/datagen.json")['datagen']
        self.resampling_dict = read_json(self.current_dir + "/configs/preprocess.json")['preprocessing'][
            "resampling"]
        self.datagen_dict_specs = self.datagen_dict['specs']
        self.datagen_dict_prep = self.datagen_dict['preprocessing']
      #  print("working init = true")

    def build(self):
        json_file = open(self.current_dir + "/model/model.json", 'r')
 #      print(self.current_dir + "/model/model.json") #print directory folder test
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.current_dir + "/model/model_checkpoint.hdf5")
        return loaded_model

    def run(self, model):  #take out info
      #  self.info = info
      #  self.case = info["case"]
      #  print ("working run = true") #test for working run function
      #  print(self.case) # no more case is alive
        arrayData = np.zeros((63, 63, 40))
        for lineTests in range(0,30):
            for lineTestp in range(-31,32):
                    for lineTestl in range(-31,32):
                        if lineTests > 8 and lineTests < 24 and lineTestp > -15 and lineTestp < 15 and lineTestl > -15 and lineTestl < 15:
                          t2_test, abk_test, zone_encoding = self.extract_patches(lineTestl*4,lineTestp*4,lineTests)
                          t2_test, abk_test = self.mean_std_standarzation(t2_test, abk_test)
                          x = [t2_test, abk_test]
                          arrayData[lineTestl+31,lineTestp+31,lineTests] = model.predict(x,verbose=1)
                        else:
                          arrayData[lineTestl+31,lineTestp+31,lineTests] = 0.0
                        iterationString = str(lineTests) + " " + str(lineTestp) + " " + str(lineTestl)
                        print(iterationString)
                        print(arrayData[lineTestl+31,lineTestp+31,lineTests])
        nrrd.write('C:/w/s/prostatecancer.ai-master/models/Densenet_T2_ABK_auc_079_nozone/OutputData.nrrd',arrayData)    
    """
        printData = ""
        for lineTestp in range(-31,32): #evry four pixels
            for lineTestl in range(-31,32):
                t2_test, abk_test, zone_encoding = self.extract_patches(lineTestl*4,lineTestp*4)
                t2_test, abk_test = self.mean_std_standarzation(t2_test, abk_test)
                x = [t2_test, abk_test, zone_encoding]
##modify from here
                printData = printData + " " + str(model.predict(x,verbose=1))  
            printData = printData + "\n"            
        print(printData)
    """
#        predicted_prob = model.predict(x, verbose=1)
#        print("successss" * 10)
#        scores = np.concatenate(predicted_prob).ravel()
#        print("predictions: {} ".format(scores))
#        description = "{:03.1f}% probability of Significant Prostate Cancer".format(scores[0] * 100)
#        response_dict = {"case": self.info["case"],
#                        "description": description,
#                         "score": str(scores[0])}
        
        #return json.dumps(response_dict)
##to here
    def resample_image(self, image, image_type):
        voxel_resampling_dict = {"t2_tse_tra": self.resampling_dict["spacing"]["t2"],
                                 "ADC": self.resampling_dict["spacing"]["dwi"],
                                 "BVAL": self.resampling_dict["spacing"]["dwi"],
                                 "Ktrans": self.resampling_dict["spacing"]["ktrans"]}

        return resample_new_spacing(image, target_spacing=voxel_resampling_dict[image_type])

    def read_image(self, image_type):
        image_paths = image_paths = ['C:/w/s/prostatecancer.ai-master/models/Densenet_T2_ABK_auc_079_nozone/Case508.nrrd'] #this samplefile
      # print(image_paths)
        assert len(image_paths) == 1, print(self.case, "more than one image or zero")
        image = sitk.ReadImage(image_paths[0])
        image = self.resample_image(image, image_type)
        image_prep = preprocess(image=image,
                                window_intensity_dict=self.datagen_dict_prep["window_intensity"],
                                zero_scale_dict=self.datagen_dict_prep["rescale_zero_one"])
        return image_prep

    def extract_patches(self,lineTestl,lineTestp,lineTests):
        zone_encoding = np.zeros((1, 3), dtype=np.float32)
        #if self.info["zone"].lower().startswith('p'):
        #    zone_encoding[0, ...] = np.array([1, 0, 0])
        #elif self.info["zone"].lower().startswith('t'):
        #    zone_encoding[0, ...] = np.array([0, 1, 0])
        #elif self.info["zone"].lower().startswith('a'):
        #    zone_encoding[0, ...] = np.array([0, 0, 1])

        zone_encoding[0, ...] = np.array([1, 0, 0]) ## fix over last lines Peripheral used for 0,50,100
        #zone_encoding[0, ...] = np.array([0, 1, 0]) ## transition needs to be tested
        #zone_encoding[0, ...] = np.array([0, 0, 1]) ## anterior needs to be tested

        size_x, size_y, size_z = self.datagen_dict_specs["output_patch_shape"]["size"]

        t2_test = np.zeros((1, size_x, size_y, size_z, 1), dtype=np.float32)
        abk_test = np.zeros((1, size_x // 2, size_y // 2, size_z, 3), dtype=np.float32)

        for enum, image_type in enumerate(['t2_tse_tra', 'ADC', 'BVAL', 'Ktrans']):
            image = self.read_image(image_type=image_type)
           # lps = self.info["lps"] #no more info
            l1 = lineTestl
            p1 = lineTestp
            s1 = lineTests
            lps = [l1,p1,s1] #lps location of click           -127 +128 L P   s=0
            ijk = image.TransformPhysicalPointToIndex(lps)
            if image_type == 't2_tse_tra':
                image_cropped = crop_roi(image, ijk, [size_x, size_y, size_z])
                image_cropped_arr = sitk.GetArrayFromImage(image_cropped)
                image_cropped_arr = np.swapaxes(image_cropped_arr, 0, 2)
                t2_test[0, ..., 0] = image_cropped_arr
            else:
                image_cropped = crop_roi(image, ijk, [size_x // 2, size_y // 2, size_z])
                image_cropped_arr = sitk.GetArrayFromImage(image_cropped)
                image_cropped_arr = np.swapaxes(image_cropped_arr, 0, 2)
                abk_test[0, ..., enum - 1] = image_cropped_arr
        return t2_test, abk_test, zone_encoding

    def mean_std_standarzation(self, t2_arr, abk_arr):
        mean_std_dir = os.path.join(os.path.dirname(__file__), "model/mean_stds/")
        t2_mean = np.load(mean_std_dir + "/training_t2_mean.npy")
        t2_std = np.load(mean_std_dir + "/training_t2_std.npy")
        #
        abk_mean = np.load(mean_std_dir + "/training_abk_mean.npy")
        abk_std = np.load(mean_std_dir + "/training_abk_std.npy")
        #
        t2_arr -= t2_mean
        t2_arr /= t2_std
        #
        abk_arr -= abk_mean
        abk_arr /= abk_std
        return t2_arr, abk_arr

def main():
    deployer = Deploy()
    model1 = deployer.build()   #build model
    model1._make_predict_function()
    deployer.run(model1)  #run the run function
    #print("working main = true")

if __name__ == "__main__":
    main()