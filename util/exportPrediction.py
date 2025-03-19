import argparse
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np




def export_Prediction_and_Activation(net, projectloader, datasetname, job_name ):

    # put network into evaluation status
    net.eval()
    # load images
    imgs = projectloader.dataset.imgs

    # load classification weights
    classification_weights = net.module._classification.weight

    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Exporting Predictions and Activations',
                    ncols=0)




    prediction_details = []
    high_level_metrics = []
    cm = np.zeros((net.module._num_classes, net.module._num_classes), dtype=int)


    pred_overview_header = ["img","prototype_id","pooled activation_score", "classification_weight class 0","classification_weight class 1","end score class 0","end score class 1","y_true","y_pred"]


    if( datasetname == "waterbirds_4_classes_eval"):

        # variables to calculate class-background group accuracies
        correct_per_group = np.zeros(4)
        count_classes = np.zeros(4, dtype=int)
        pred_overview_header.append("background_group")

    if( datasetname == "celebA"):
        attributes_df = pd.read_csv('./data/CELEB_A/celebA/list_attr_celeba.csv',index_col=0)
        # add all attribute names to header
        for col in attributes_df.columns:
            pred_overview_header.append(col)

    if (datasetname == "isic" or datasetname == "B_insert_p" or datasetname == "mal_insert" or datasetname == "only_B_w_P" ):

        isic_metadata = pd.read_csv('./data/ISIC_ALL/metadata.csv', index_col=0)
        pred_overview_header.append("patch")


    # go through each image one by one
    for i, (x, y) in img_iter:

        # create new lines for export dataframes
        new_pred_overview_line = []

        # new list for activated prototypes per image
        activated_prototypes = []

        # extract image filename
        img = imgs[i][0].split("/")[-1]

        # forward input through network and save prediction results
        _, pooled, out = net(x, inference=True)
        max_out_scores, y_pred = torch.max(out, dim=1)
        y_pred = y_pred.item()
        # save ground-truth label
        y_true = y.item()

        #add prototypes with activation scores > 0
        activated_prototypes.append(torch.cat((torch.nonzero(pooled[0]), pooled[0][torch.nonzero(pooled[0])]), 1),)

        if (datasetname == 'waterbirds_4_classes_eval'):

            background_group = y_true
            np.add.at(count_classes, background_group, 1)
            y_true = torch.from_numpy(np.where((y_true == 0) | (y_true == 1), 0, 1)).item()

            if(y_true == y_pred):
                correct_per_group[background_group] += 1

        cm[y_true][y_pred] += 1

        #create new row entry

        for prototype_id,activation in activated_prototypes[0]:
            new_pred_overview_line = [img, int(prototype_id.item()), activation.item(), classification_weights[0,int(prototype_id.item())].item(),classification_weights[1,int(prototype_id.item())].item(),out[0][0].item(),out[0][1].item(), y_true, y_pred]

            if (datasetname == 'waterbirds_4_classes_eval'):
                new_pred_overview_line.append(background_group)




            if (datasetname == "isic" or datasetname == "B_insert_p" or datasetname == "mal_insert" or datasetname == "only_B_w_P" ):
                img = img.replace("_2.jpg",".jpg")
                patch = isic_metadata.loc[img]["patches"]
                new_pred_overview_line.append(patch)

            if (datasetname == "celebA"):
                # add all attribute values of img to new line
                for attribute in attributes_df.loc[img]:
                    new_pred_overview_line.append(attribute)

            prediction_details.append(new_pred_overview_line)




    overall_acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]) if (cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])>0 else "n.a."
    class_0_acc = cm[0][0]/(cm[0][1]+cm[0][0]) if (cm[0][1]+cm[0][0]) > 0 else "n.a."
    class_1_acc = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0]+cm[1][1]) > 0 else "n.a."

    acc_overview_header = ["datasetname","overall_acc", "class_0_acc", "class_1_acc"]
    high_level_metrics_line = [datasetname, overall_acc, class_0_acc, class_1_acc]


    if ( datasetname == "waterbirds_4_classes_eval"):

        acc_lb_l = correct_per_group[0]/count_classes[0]
        acc_lb_w = correct_per_group[1]/count_classes[1]
        acc_wb_l = correct_per_group[2]/count_classes[2]
        acc_wb_w = correct_per_group[3]/count_classes[3]
        acc_overview_header.append("acc lb-l")
        acc_overview_header.append("acc lb-w")

        acc_overview_header.append("acc wb-l")
        acc_overview_header.append("acc wb-w")
        high_level_metrics_line.append(acc_lb_l)
        high_level_metrics_line.append(acc_lb_w)
        high_level_metrics_line.append(acc_wb_l)
        high_level_metrics_line.append(acc_wb_w)

    high_level_metrics.append(high_level_metrics_line)
    export_low_level_df = pd.DataFrame(prediction_details, columns=pred_overview_header)
    export_low_level_df.to_csv('./data/CSV_EXPORTS/prediction_overview_' + job_name + '.csv')

    export_acc_df = pd.DataFrame(high_level_metrics, columns=acc_overview_header)
    export_acc_df.to_csv('./data/CSV_EXPORTS/acc_overview_'+job_name+'.csv')


