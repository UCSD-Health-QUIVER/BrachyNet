# -*- encoding: utf-8 -*-
import torch
import os
import sklearn.model_selection
import numpy as np
import torch.nn as nn
import pandas as pd
import custom_loss
import torchio as tio



def online_evaluation(trainer):
    list_Dose_score = []
    list_Dose_score_TO = []
    list_Dose_score_TON = []
    list_Dose_score_TR = []
    list_Dose_score_TRN = []
    for batch_idx, list_loader_output in enumerate(trainer.setting.val_loader):
        with torch.no_grad():
            trainer.setting.network.eval()
            input_, gt_dose= trainer.prepare_batch(list_loader_output)
            input_ = input_.to(trainer.setting.device)
            [_,prediction_B] = trainer.setting.network(input_)
            pred_B = prediction_B.cpu().data[0, :, :, :, :]          
            gt_dose = gt_dose[0].squeeze(0)

            mse_loss = nn.MSELoss(reduction="none")
            loss = mse_loss(pred_B,gt_dose)

            Dose_score = torch.mean(loss)
            list_Dose_score.append(Dose_score.item())

            # switch case for applicator type
            if list_loader_output['patient_id'][0].split("_")[-1] == "T&O":
                list_Dose_score_TO.append(Dose_score.item())
            elif list_loader_output['patient_id'][0].split("_")[-1] == "T&ON":
                list_Dose_score_TON.append(Dose_score.item())
            elif list_loader_output['patient_id'][0].split("_")[-1] == "T&R":
                list_Dose_score_TR.append(Dose_score.item())
            elif list_loader_output['patient_id'][0].split("_")[-1] == "T&RN":
                list_Dose_score_TRN.append(Dose_score.item()) 


            try:
                trainer.print_log_to_file('========> ' + str(list_loader_output['patient_id'][0]) + ':  ' + str(Dose_score.item()), 'a')
                
            except Exception as e:
                with open(trainer.setting.output_dir + '/errors.txt', 'a', encoding="utf-8") as file:
                    file.write(str(e))

    try:
        trainer.print_log_to_file('===============================================> Average MSE Dose Val (percent Rx): '
                                  + str(np.mean(list_Dose_score)), 'a')
                                
    except Exception as e:
        with open(trainer.setting.output_dir + '/errors.txt', 'a', encoding="utf-8") as file:
            file.write(str(e))
    
    
    # in case there are no patients of a certain applicator type
    if len(list_Dose_score_TO) == 0:
        list_Dose_score_TO.append(0)
    if len(list_Dose_score_TON) == 0:
        list_Dose_score_TON.append(0)
    if len(list_Dose_score_TR) == 0:
        list_Dose_score_TR.append(0)
    if len(list_Dose_score_TRN) == 0:
        list_Dose_score_TRN.append(0)


    trainer.log.val_curve.append(np.mean(list_Dose_score))
    trainer.log.train_curve.append(trainer.log.average_train_loss)
    trainer.log.val_curve_TO.append(np.mean(list_Dose_score_TO))
    trainer.log.val_curve_TON.append(np.mean(list_Dose_score_TON))
    trainer.log.val_curve_TR.append(np.mean(list_Dose_score_TR))
    trainer.log.val_curve_TRN.append(np.mean(list_Dose_score_TRN))


    csv_dir = trainer.setting.output_dir
    df = pd.DataFrame({"train":trainer.log.train_curve, "val":trainer.log.val_curve, "TO":trainer.log.val_curve_TO, "TON":trainer.log.val_curve_TON, "TR":trainer.log.val_curve_TR, "TRN":trainer.log.val_curve_TRN })
    df.to_csv(os.path.join(csv_dir,"training_curves.csv"))
    

    return np.mean(list_Dose_score)

