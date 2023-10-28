# # -*- encoding: utf-8 -*-
from ast import Raise
from pickletools import uint8
import torch.utils.data as data
import SimpleITK as sitk
import numpy as np
import random
from math import degrees
import sklearn.model_selection
import torch
import torchio as tio
import os
import json


"""
The following functions are used to load the data. The data is loaded using TorchIO, which is a library for loading medical images.
The data is loaded as a tio.Subject, which is a dictionary of tio.ScalarImage and tio.LabelMap. The tio.ScalarImage is used for the dose, and the tio.LabelMap is used for the structures.
This file is used for infrence and evaluation (GeneratePredictions_*), and is not used for training.
"""




def get_subject_list(patients_dir, applicator = None):
    """
    This function is used to get the list of subjects for training and validation.
    :param patients_dir: the directory of the patients, usually the path is like: /home/brachytherapy/Brachytherapy_Modeling/Dataset_N/Test
    :param applicator: the applicator type, if None, all applicators will be included, otherwise, only the specified applicator will be included
    :return: a list of subjects (tio.Subject)
    """
    patient_list = []
    banned = [] # exclude patients with bad data, if any
    try:
        os.removedirs(os.path.join(patients_dir,"zeros.nii")) # remove zeros.nii if it exists, in case it was a different size
    except:
        pass
    zeros = np.zeros((128,128,128))
    zeros_img = sitk.GetImageFromArray(zeros)
    sitk.WriteImage(zeros_img,os.path.join(patients_dir,"zeros.nii")) # create zeros.nii if it does not exist
 
    for patient_folder in os.listdir(patients_dir):
        if patient_folder in banned: # no need to load these patients
            continue
        if applicator is not None: # if applicator is specified, only load patients with that applicator
            if patient_folder.split("_")[-1] != applicator:
                continue
        # here, we chose to exclude some dwell positions from the ring applicator, because they are not used in the treatment
        exclude_ring_dps = []
        if not os.path.exists(os.path.join(patients_dir,patient_folder,"ring_updated.nii")) and (patient_folder.split("_")[-1] == "T&R" or patient_folder.split("_")[-1] == "T&RN"):
            with open(os.path.join(patients_dir,patient_folder,'dwell_positions_map.txt')) as f:
                dwell_positions_data = json.load(f)
            with open(os.path.join(patients_dir,patient_folder,'dwell_times.txt')) as f:
                dwell_times_data = f.readlines()
            dwell_times_data = [float(x.strip()) for x in dwell_times_data]


            ring_dps = dwell_positions_data[list(dwell_positions_data)[1]]
            for dp in ring_dps:
                if dwell_times_data[dp] < 0.5:
                    exclude_ring_dps.append(dp)
            ring_image = None
            for dp in ring_dps:
                if dp not in exclude_ring_dps:
                    if ring_image is None:
                        ring_image = sitk.ReadImage(os.path.join(patients_dir,patient_folder,"dp{}.nii.gz".format(dp)))
                    else:
                        temp_image = sitk.ReadImage(os.path.join(patients_dir,patient_folder,"dp{}.nii.gz".format(dp)))
                        ring_image += temp_image
            ring_image *= 1000 
            sitk.WriteImage(ring_image,os.path.join(patients_dir,patient_folder,"ring_updated.nii"))
       




        full_dir = os.path.join(patients_dir,patient_folder)

        list_structures = ['true_dose_upsample',
                    'dwell_positions',
                    'hrctv',
                    'bladder',
                    'rectum',
                    'sigmoid',
                    'tandem',
                    'ovoid',
                    'ring_updated',
                    'needle'
                    ] # all models use these structures


        subject_dict = {} # this is the dictionary that will be used to create the tio.Subject

        for structure_name in list_structures:
            if os.path.exists(os.path.join(full_dir,structure_name+".nii")):
                if structure_name in ["true_dose_upsample","tandem","ovoid","ring_updated","needle"]:
                    # load to RAM, i.e. not lazy loading, this consumes a lot of memory
                    # image = sitk.ReadImage(os.path.join(full_dir,structure_name+".nii.gz"))
                    # array = sitk.GetArrayFromImage(image)
                    # tensor = torch.from_numpy(np.expand_dims(array,0))
                    # subject_dict[structure_name]=tio.ScalarImage(tensor = tensor)


                    # load from hard disk when needed, i.e. lazy loading    
                    subject_dict[structure_name]=tio.ScalarImage(os.path.join(full_dir,structure_name+".nii"))


                elif structure_name in ["hrctv","bladder","rectum","sigmoid", "dwell_positions"]:
                    #load to RAM
                    # image = sitk.ReadImage(os.path.join(full_dir,structure_name+".nii.gz"))
                    # array = sitk.GetArrayFromImage(image)
                    # tensor = torch.from_numpy(np.expand_dims(array,0))
                    # subject_dict[structure_name]=tio.LabelMap(tensor = tensor)

                    # from hard disk
                    subject_dict[structure_name]=tio.LabelMap(os.path.join(full_dir,structure_name+".nii")) 
                    # the ScalarImage vs. Label map control which type of interpolation is used when resampling, i.e. nearest neighbor vs. linear
            

                        
            else:

                subject_dict[structure_name] = tio.LabelMap(os.path.join(patients_dir,"zeros.nii")) ## path not exist, use zeros.nii instead
                # This is used for channels not used in a particular patient. i.e. ring for a tandem and ovoid patient


            
        subject_dict["patient_id"] = patient_folder
        subject = tio.Subject(subject_dict)
        patient_list.append(subject)
    return patient_list

def get_loader(train_bs=1, val_bs=1, train_num_samples_per_epoch=1, val_num_samples_per_epoch=1, num_works=0, slurm_dir=None, applicator=None, training = False):

    if slurm_dir is None: # when running locally, use the hardcoded path
        data_dir_train = r'' # hardcoded path to training data on local machine for QA
        data_dir_val = r''
    else: # when running on slurm, use the slurm_dir
        data_dir_train = os.path.join(slurm_dir,"Train")
        data_dir_val = os.path.join(slurm_dir,"Validation")

    train_subjects = get_subject_list(data_dir_train,applicator)
    val_subjects = get_subject_list(data_dir_val, applicator)

    if training:
        transforms = tio.Compose([

        tio.RandomFlip(p=0.8,axes=(0,1,2)),

        ])
        train_dataset = tio.SubjectsDataset(train_subjects,transform=transforms)
    else:

        train_dataset = tio.SubjectsDataset(train_subjects)
  
    val_dataset = tio.SubjectsDataset(val_subjects)



    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_works)
                                   #pin_memory=False,collate_fn=tio.utils.history_collate) # this was used for debugging, not sure if it is needed
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works)
                                 #pin_memory=False,collate_fn=tio.utils.history_collate)
   

 
    return train_loader, val_loader 

def get_test_loader(train_bs=1, val_bs=1, train_num_samples_per_epoch=1, val_num_samples_per_epoch=1, num_works=0, slurm_dir=None, applicator = None):

    if slurm_dir is None:
        data_dir_test = r'' # hardcoded path to training data on local machine for QA
    else:
        data_dir_test = os.path.join(slurm_dir,"Test")

    test_subjects = get_subject_list(data_dir_test, applicator)
    test_dataset = tio.SubjectsDataset(test_subjects)

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works)


    return test_loader 
