# -*- encoding: utf-8 -*-
from genericpath import exists
import os
import sys
import argparse
import gc
from model import *
from network_trainer import *
import itertools
import numpy as np
import SimpleITK as sitk
import TorchIODataloader as evalloader
import matplotlib.pyplot as plt
import json



if __name__ == "__main__":
    print("running")

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_id', type=int, default=[0],
                        help='GPU id used for testing (default: 0)')
    parser.add_argument('--model_path', type=str, default= os.path.join(os.path.dirname(os.path.realpath(__file__)),os.path.join('BrachyNet_','best_val_evaluation_index.pkl')))
    parser.add_argument('--TTA', type=bool, default=False,
                        help='do test-time augmentation, default True')
    parser.add_argument('--slurm_dir', type=str,default='/scratch/$USER/job_$SLURM_JOBID')   
    args = parser.parse_args()
    slurm_dir = args.slurm_dir
    
    
    if os.name == 'nt':
        slurm_dir = None
            
    model_path = args.model_path

    ### model path is directory to checkpoint file
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'Alpha'


    trainer.setting.output_dir = os.path.join(os.path.dirname(model_path),'Predictions')



    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)

    trainer.setting.network = Model(in_ch=9, out_ch=1,
                                    list_ch_A=[-1,16, 32, 64, 128, 256],
                                    list_ch_B=[-1,16, 32, 64, 128, 256],
                                    dropout_prob=0.0
                                    ).float()
    # load the saved model
    trainer.init_trainer(ckpt_file=model_path,
                list_GPU_ids=[0],
                only_network=True)

    # Start inference
    print('\n\n# Start inference !')

    applicator_type = None # change for specific applicator type
    test = evalloader.get_test_loader(slurm_dir=slurm_dir, applicator=applicator_type, training=False)

    combined_dose = None
    combined_pred = None
    count = 0
    TTA = True
    plots_str = "Plots"
    if TTA:
        plots_str = "Plots_TTA"
    if not os.path.exists(os.path.join(os.path.dirname(model_path),plots_str)):
        os.makedirs(os.path.join(os.path.dirname(model_path),plots_str))
    plots_dir = os.path.join(os.path.dirname(model_path),plots_str)
    
    fold = "Test"

    # TODO: wrap this in a function
    if not os.path.exists(os.path.join(trainer.setting.output_dir,fold)):
        os.mkdir(os.path.join(trainer.setting.output_dir,fold))
    val_dir = os.path.join(trainer.setting.output_dir,fold)
    list_loaders = []
    dose_arrays, dict_volumes, predicted_dose_arrays = {}, {}, {}
    for batch_idx, list_loader_output in enumerate(test):
    

        # direction = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0) # this is the clincal direction
        with torch.no_grad():
            trainer.setting.network.eval()
            input_, gt_dose= trainer.prepare_batch(list_loader_output)



     
            if "zero" in list_loader_output['patient_id'][0]: # skip the zeros.nii file
                continue
            if not os.path.exists(os.path.join(val_dir,list_loader_output['patient_id'][0])):
                os.mkdir(os.path.join(val_dir,list_loader_output['patient_id'][0]))

            full_pat_dir = os.path.join(val_dir,list_loader_output['patient_id'][0])


            [_,prediction_B] = trainer.setting.network(input_.to(device=trainer.setting.device))

            if TTA:
                flipped_input_2 = torch.flip(input_, [2])
                [_,prediction_flipped_input_2] = trainer.setting.network(flipped_input_2.to(device=trainer.setting.device))
                prediction_flipped_input_2 = torch.flip(prediction_flipped_input_2, [2])

                flipped_input_3 = torch.flip(input_, [3])
                [_,prediction_flipped_input_3] = trainer.setting.network(flipped_input_3.to(device=trainer.setting.device))
                prediction_flipped_input_3 = torch.flip(prediction_flipped_input_3, [3])

                flipped_input_4 = torch.flip(input_, [4])
                [_,prediction_flipped_input_4] = trainer.setting.network(flipped_input_4.to(device=trainer.setting.device))
                prediction_flipped_input_4 = torch.flip(prediction_flipped_input_4, [4])

                flipped_input_5 = torch.flip(input_, [2,3])
                [_,prediction_flipped_input_5] = trainer.setting.network(flipped_input_5.to(device=trainer.setting.device))
                prediction_flipped_input_5 = torch.flip(prediction_flipped_input_5, [2,3])

                flipped_input_6 = torch.flip(input_, [2,4])
                [_,prediction_flipped_input_6] = trainer.setting.network(flipped_input_6.to(device=trainer.setting.device))
                prediction_flipped_input_6 = torch.flip(prediction_flipped_input_6, [2,4])

                flipped_input_7 = torch.flip(input_, [3,4])
                [_,prediction_flipped_input_7] = trainer.setting.network(flipped_input_7.to(device=trainer.setting.device))
                prediction_flipped_input_7 = torch.flip(prediction_flipped_input_7, [3,4])

                flipped_input_8 = torch.flip(input_, [2,3,4])
                [_,prediction_flipped_input_8] = trainer.setting.network(flipped_input_8.to(device=trainer.setting.device))
                prediction_flipped_input_8 = torch.flip(prediction_flipped_input_8, [2,3,4])

                prediction_B = (prediction_B + prediction_flipped_input_2 + prediction_flipped_input_3 + prediction_flipped_input_4 + prediction_flipped_input_5 + prediction_flipped_input_6 + prediction_flipped_input_7 + prediction_flipped_input_8)/8

            

            prediction_B = prediction_B.cpu().data[:, :, :, :, :]
    


            prediction_B = prediction_B.squeeze()
            

            prediction_B = torch.swapaxes(prediction_B, 0, 2)

            x_nii = sitk.GetImageFromArray(prediction_B.squeeze())

            sitk.WriteImage(x_nii, os.path.join(full_pat_dir, "prediction.nii.gz"))

            gt_dose[0] = gt_dose[0].squeeze()
            gt_dose[0] = torch.swapaxes(gt_dose[0], 0, 2)
            y_nii = sitk.GetImageFromArray(gt_dose[0].squeeze())


            sitk.WriteImage(y_nii, os.path.join(full_pat_dir, "true_dose.nii.gz"))
            
            
            diff = gt_dose[0].squeeze() - prediction_B.squeeze()

            diff = sitk.GetImageFromArray(diff)

            sitk.WriteImage(diff, os.path.join(full_pat_dir, "Actual_Minus_Pred.nii.gz"))

    fold = "Train"
    train, val = evalloader.get_loader(slurm_dir=slurm_dir, applicator=applicator_type)

    if not os.path.exists(os.path.join(trainer.setting.output_dir,fold)):
        os.mkdir(os.path.join(trainer.setting.output_dir,fold))
    val_dir = os.path.join(trainer.setting.output_dir,fold)
    list_loaders = []
    dose_arrays, dict_volumes, predicted_dose_arrays = {}, {}, {}
    for batch_idx, list_loader_output in enumerate(train):
    


        with torch.no_grad():
            trainer.setting.network.eval()
            input_, gt_dose= trainer.prepare_batch(list_loader_output)



    
            if "zero" in list_loader_output['patient_id'][0]:
                continue
            if not os.path.exists(os.path.join(val_dir,list_loader_output['patient_id'][0])):
                os.mkdir(os.path.join(val_dir,list_loader_output['patient_id'][0]))
            # else:
            #     continue
            full_pat_dir = os.path.join(val_dir,list_loader_output['patient_id'][0])


            [_,prediction_B] = trainer.setting.network(input_.to(device=trainer.setting.device))

            if TTA:
                flipped_input_2 = torch.flip(input_, [2])
                [_,prediction_flipped_input_2] = trainer.setting.network(flipped_input_2.to(device=trainer.setting.device))
                prediction_flipped_input_2 = torch.flip(prediction_flipped_input_2, [2])

                flipped_input_3 = torch.flip(input_, [3])
                [_,prediction_flipped_input_3] = trainer.setting.network(flipped_input_3.to(device=trainer.setting.device))
                prediction_flipped_input_3 = torch.flip(prediction_flipped_input_3, [3])

                flipped_input_4 = torch.flip(input_, [4])
                [_,prediction_flipped_input_4] = trainer.setting.network(flipped_input_4.to(device=trainer.setting.device))
                prediction_flipped_input_4 = torch.flip(prediction_flipped_input_4, [4])

                flipped_input_5 = torch.flip(input_, [2,3])
                [_,prediction_flipped_input_5] = trainer.setting.network(flipped_input_5.to(device=trainer.setting.device))
                prediction_flipped_input_5 = torch.flip(prediction_flipped_input_5, [2,3])

                flipped_input_6 = torch.flip(input_, [2,4])
                [_,prediction_flipped_input_6] = trainer.setting.network(flipped_input_6.to(device=trainer.setting.device))
                prediction_flipped_input_6 = torch.flip(prediction_flipped_input_6, [2,4])

                flipped_input_7 = torch.flip(input_, [3,4])
                [_,prediction_flipped_input_7] = trainer.setting.network(flipped_input_7.to(device=trainer.setting.device))
                prediction_flipped_input_7 = torch.flip(prediction_flipped_input_7, [3,4])

                flipped_input_8 = torch.flip(input_, [2,3,4])
                [_,prediction_flipped_input_8] = trainer.setting.network(flipped_input_8.to(device=trainer.setting.device))
                prediction_flipped_input_8 = torch.flip(prediction_flipped_input_8, [2,3,4])

                prediction_B = (prediction_B + prediction_flipped_input_2 + prediction_flipped_input_3 + prediction_flipped_input_4 + prediction_flipped_input_5 + prediction_flipped_input_6 + prediction_flipped_input_7 + prediction_flipped_input_8)/8

            

            prediction_B = prediction_B.cpu().data[:, :, :, :, :]
    


            prediction_B = prediction_B.squeeze()
            

            prediction_B = torch.swapaxes(prediction_B, 0, 2)

            x_nii = sitk.GetImageFromArray(prediction_B.squeeze())

            sitk.WriteImage(x_nii, os.path.join(full_pat_dir, "prediction.nii.gz"))

            gt_dose[0] = gt_dose[0].squeeze()
            gt_dose[0] = torch.swapaxes(gt_dose[0], 0, 2)
            y_nii = sitk.GetImageFromArray(gt_dose[0].squeeze())


            sitk.WriteImage(y_nii, os.path.join(full_pat_dir, "true_dose.nii.gz"))
            
            
            diff = gt_dose[0].squeeze() - prediction_B.squeeze()

            diff = sitk.GetImageFromArray(diff)

            sitk.WriteImage(diff, os.path.join(full_pat_dir, "Actual_Minus_Pred.nii.gz"))
    
    
    
    fold = "Validation"
    if not os.path.exists(os.path.join(trainer.setting.output_dir,fold)):
        os.mkdir(os.path.join(trainer.setting.output_dir,fold))
    val_dir = os.path.join(trainer.setting.output_dir,fold)

    dose_arrays, dict_volumes, predicted_dose_arrays = {}, {}, {}
    for batch_idx, list_loader_output in enumerate(val):
    
        with torch.no_grad():
            trainer.setting.network.eval()
            input_, gt_dose= trainer.prepare_batch(list_loader_output)



    
            if "zero" in list_loader_output['patient_id'][0]:
                continue
            if not os.path.exists(os.path.join(val_dir,list_loader_output['patient_id'][0])):
                os.mkdir(os.path.join(val_dir,list_loader_output['patient_id'][0]))

            full_pat_dir = os.path.join(val_dir,list_loader_output['patient_id'][0])


            [_,prediction_B] = trainer.setting.network(input_.to(device=trainer.setting.device))

            if TTA:
                flipped_input_2 = torch.flip(input_, [2])
                [_,prediction_flipped_input_2] = trainer.setting.network(flipped_input_2.to(device=trainer.setting.device))
                prediction_flipped_input_2 = torch.flip(prediction_flipped_input_2, [2])

                flipped_input_3 = torch.flip(input_, [3])
                [_,prediction_flipped_input_3] = trainer.setting.network(flipped_input_3.to(device=trainer.setting.device))
                prediction_flipped_input_3 = torch.flip(prediction_flipped_input_3, [3])

                flipped_input_4 = torch.flip(input_, [4])
                [_,prediction_flipped_input_4] = trainer.setting.network(flipped_input_4.to(device=trainer.setting.device))
                prediction_flipped_input_4 = torch.flip(prediction_flipped_input_4, [4])

                flipped_input_5 = torch.flip(input_, [2,3])
                [_,prediction_flipped_input_5] = trainer.setting.network(flipped_input_5.to(device=trainer.setting.device))
                prediction_flipped_input_5 = torch.flip(prediction_flipped_input_5, [2,3])

                flipped_input_6 = torch.flip(input_, [2,4])
                [_,prediction_flipped_input_6] = trainer.setting.network(flipped_input_6.to(device=trainer.setting.device))
                prediction_flipped_input_6 = torch.flip(prediction_flipped_input_6, [2,4])

                flipped_input_7 = torch.flip(input_, [3,4])
                [_,prediction_flipped_input_7] = trainer.setting.network(flipped_input_7.to(device=trainer.setting.device))
                prediction_flipped_input_7 = torch.flip(prediction_flipped_input_7, [3,4])

                flipped_input_8 = torch.flip(input_, [2,3,4])
                [_,prediction_flipped_input_8] = trainer.setting.network(flipped_input_8.to(device=trainer.setting.device))
                prediction_flipped_input_8 = torch.flip(prediction_flipped_input_8, [2,3,4])

                prediction_B = (prediction_B + prediction_flipped_input_2 + prediction_flipped_input_3 + prediction_flipped_input_4 + prediction_flipped_input_5 + prediction_flipped_input_6 + prediction_flipped_input_7 + prediction_flipped_input_8)/8

            

            prediction_B = prediction_B.cpu().data[:, :, :, :, :]
    


            prediction_B = prediction_B.squeeze()
            

            prediction_B = torch.swapaxes(prediction_B, 0, 2)

            x_nii = sitk.GetImageFromArray(prediction_B.squeeze())

            sitk.WriteImage(x_nii, os.path.join(full_pat_dir, "prediction.nii.gz"))

            gt_dose[0] = gt_dose[0].squeeze()
            gt_dose[0] = torch.swapaxes(gt_dose[0], 0, 2)
            y_nii = sitk.GetImageFromArray(gt_dose[0].squeeze())


            sitk.WriteImage(y_nii, os.path.join(full_pat_dir, "true_dose.nii.gz"))
            
            
            diff = gt_dose[0].squeeze() - prediction_B.squeeze()

            diff = sitk.GetImageFromArray(diff)

            sitk.WriteImage(diff, os.path.join(full_pat_dir, "Actual_Minus_Pred.nii.gz"))
      
    