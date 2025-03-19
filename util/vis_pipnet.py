from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
import torch.utils.data
import os
from PIL import Image, ImageDraw as D
import torchvision.transforms as transforms
import torchvision
from util.func import get_patch_size
import random
import pandas as pd
import numpy as np

from util.shortcuts.rm_shortcuts_manually import removeShortcutsManually


@torch.no_grad()
def visualize_topk(net, projectloader, num_classes, device, foldername, args: argparse.Namespace, k=10):
    print("Visualizing prototypes for topk...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)
    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        tensors_per_prototype[p]=[]
    
    patchsize, skip = get_patch_size(args)

    imgs = projectloader.dataset.imgs
    
    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight

    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,
                    desc='Collecting topk',
                    ncols=0)

    # Iterate through the data
    images_seen = 0
    topks = dict()
    # Iterate through the training set
    for i, (xs, ys) in img_iter:

        images_seen+=1
        xs, ys = xs.to(device), ys.to(device)

        with torch.no_grad():
            # Use the model to classify this batch of input data
            pfs, pooled, _ = net(xs, inference=True)
            pooled = pooled.squeeze(0) 
            pfs = pfs.squeeze(0)
            #ids = np.array([713, 106, 206, 598, 699])

            for p in range(pooled.shape[0]):
            #for p in ids:
                c_weight = torch.max(classification_weights[:,p])
                if c_weight > 1e-3:#ignore prototypes that are not relevant to any class
                    if p not in topks.keys():
                        topks[p] = []
                    if len(topks[p]) < k:
                        topks[p].append((i, pooled[p].item()))
                    else:
                        topks[p] = sorted(topks[p], key=lambda tup: tup[1], reverse=True)
                        if topks[p][-1][1] < pooled[p].item():
                            topks[p][-1] = (i, pooled[p].item())
                        if topks[p][-1][1] == pooled[p].item():
                            # equal scores. randomly chose one (since dataset is not shuffled so latter images with same scores can now also get in topk).
                            replace_choice = random.choice([0, 1])
                            if replace_choice > 0:
                                topks[p][-1] = (i, pooled[p].item())
    '''
    shortcut_candidates = []
    if args.dataset == 'isic':
        isic_metadata = pd.read_csv("./data/ISIC_ALL/metadata.csv", header=0, index_col=0)

    for p in topks.keys():
        for i in topks[p]:
            if (args.dataset == 'isic'):
                segment_path = "./data/ISIC_ALL/segmentation"

                with torch.no_grad():
                    softmaxes, pooled, out = net(xs, inference=True)  # softmaxes has shape (1, num_prototypes, W, H)
                    outmax = torch.amax(out, dim=1)[0]  # shape ([1]) because batch size of projectloader is 1
                    if outmax.item() == 0.:
                        abstained += 1

                # Take the max per prototype.                             
                max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
                max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
                max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h,
                                                                         dim=1)  # shape (num_prototypes)

                c_weight = torch.max(
                    classification_weights[:, p])  # ignore prototypes that are not relevant to any class
                if (c_weight > 1e-10) or ('pretrain' in foldername):

                    h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                    w_idx = max_idx_per_prototype_w[p]

                    img_to_open = imgs[i]

                    # load segmentation:

                    if isinstance(img_to_open, tuple) or isinstance(img_to_open,
                                                                    list):  # dataset contains tuples of (img,label)
                        img_to_open = img_to_open[0]
                    image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open))
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0)  # shape (1, 3, h, w)
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size,
                                                                                         softmaxes.shape, patchsize,
                                                                                         skip, h_idx, w_idx)

                img_filename = img_to_open.split("/")[-1]
                
                if (img_filename in np.array(isic_metadata.loc[isic_metadata["patches"] == 1].index)):
                    print("eins mit patch gefunden")
                if (isic_metadata.loc[img_filename]["patches"] == 1):
                    segment_opened = Image.open(os.path.join(segment_path, img_filename))
                    segment_resized = transforms.Resize(size=(args.image_size, args.image_size))(segment_opened)
                    segment_rgb_converted = segment_resized.convert('RGB')
    
                    segment_rgb_converted = np.array((segment_rgb_converted))
                    seg_tensor_patch = segment_rgb_converted[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    seg_tensor_patch_avg = seg_tensor_patch.mean(dim=0)
                    if (seg_tensor_patch_avg.item() >= 0):
                        shortcut_candidates.append(p)
    
                    elif (p in shortcut_candidates):
                        shortcut_candidates.remove(p)''' #Kann ich shortcut discovery part auch hier machen ?
    if args.dataset == 'isic':
        isic_metadata = pd.read_csv("./data/ISIC_ALL/metadata.csv", header=0, index_col=0)

    alli = []
    prototypes_not_used = []
    for p in topks.keys():
        found = False
        for idx, score in topks[p]:
            alli.append(idx)
            if score > 0.1:  #in case prototypes have fewer than k well-related patches
                found = True
        if not found:
            prototypes_not_used.append(p)



    print(len(prototypes_not_used), "prototypes do not have any similarity score > 0.1. Will be ignored in visualisation.")
    abstained = 0
    segment_path = ""


    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=50.,

                    desc='Visualizing topk',
                    ncols=0)

    shortcut_for_imgs = {}
    shortcut_candidates = []
    shortcut_candidates_would_be_removed = []
    shortcuts_under_weight = []
    shortcut_counter = {}
    for p in topks.keys():
        shortcut_counter[p] = 0
    for i, (xs, ys) in img_iter:

        #shuffle is false so should lead to same order as in imgs
        # hier mal testen was passiert wenn alli nicht geprüft wird, also jedes i genommen wird, kmmen wir dann zu jedem Bild das wir wollen? @TODO

        if i in alli:

            # explain what happens here. The images @TODO

            xs, ys = xs.to(device), ys.to(device)
            for p in topks.keys():
                if p not in prototypes_not_used:
                    for idx, score in topks[p]:
                        if idx == i:
                            # Use the model to classify this batch of input data
                            with torch.no_grad():
                                softmaxes, pooled, out = net(xs, inference=True) #softmaxes has shape (1, num_prototypes, W, H)
                                outmax = torch.amax(out,dim=1)[0] #shape ([1]) because batch size of projectloader is 1
                                if outmax.item() == 0.:
                                    abstained+=1

                            # Take the max per prototype.                             
                            max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
                            max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
                            max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) #shape (num_prototypes)
                            
                            c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class

                            if (c_weight > 1e-10) or ('pretrain' in foldername):
                                
                                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                                w_idx = max_idx_per_prototype_w[p]
                                
                                img_to_open = imgs[i]

                                # load segmentation:

                                if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                                    img_to_open = img_to_open[0]
                                image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open))
                                img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                                h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)
                                img_tensor_patch = img_tensor[:, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                                #image.save(os.path.join(segment_path,img_to_open))
                                #@TODO
                                #if(args.detectShortcuts):
                                if( args.dataset == 'isic' ):
                                    print(topks[p])
                                    segment_path = "./data/ISIC_ALL/segmentation"
                                    img_filename = img_to_open.split("/")[-1]
                                    if(img_filename in np.array(isic_metadata.loc[isic_metadata["patches"]==1].index)):
                                            print("eins mit patch gefunden")
                                    if(isic_metadata.loc[img_filename]["patches"]==1):
                                        segment_opened = Image.open(os.path.join(segment_path, img_filename))
                                        segment_resized = transforms.Resize(size=(args.image_size, args.image_size))(segment_opened)
                                        segment_rgb_converted = segment_resized.convert('RGB')


                                        segment_rgb_converted = np.array((segment_rgb_converted))
                                        seg_tensor_patch = segment_rgb_converted[h_coor_min:h_coor_max, w_coor_min:w_coor_max, :]

                                        seg_tensor_patch_avg = seg_tensor_patch.mean()
                                        if (seg_tensor_patch_avg >= 55):
                                            shortcut_candidates.append(p)

                                        elif(p in shortcut_candidates):
                                                shortcut_candidates.remove(p)

                                if (args.dataset == 'waterbirds_4_classes_eval'):
                                    segment_path = "./data/WATERBIRD_ALL/segmentations"
                                    img_filename = img_to_open.split("/")[-1].strip(".jpg")
                                    img_filename = img_filename + ".png"

                                    segment_opened = Image.open(os.path.join(segment_path, img_filename))
                                    segment_resized = transforms.Resize(size=(args.image_size, args.image_size))(
                                        segment_opened)
                                    segment_rgb_converted = segment_resized.convert('RGB')

                                    segment_rgb_converted = np.array((segment_rgb_converted))
                                    seg_tensor_patch = segment_rgb_converted[
                                                       h_coor_min:h_coor_max,
                                                       w_coor_min:w_coor_max,:]

                                    seg_tensor_patch_avg = seg_tensor_patch.mean()
                                    if (seg_tensor_patch_avg.item() == 0):^
                                        if(p not in shortcut_candidates):
                                            shortcut_candidates.append(p)
                                            shortcut_for_imgs[p] = [i,img_filename]

                                        shortcut_counter[p] += 1

                                    elif (seg_tensor_patch_avg.item() > 0) & (p in shortcut_candidates):
                                        shortcut_candidates_would_be_removed.append(p)


                                saved[p]+=1
                                tensors_per_prototype[p].append(img_tensor_patch)
                            else:
                                shortcuts_under_weight.append(p)
    shortcuts_counter_gr_5 = [[key,value] for key, value in shortcut_counter.items() if value >= 5]
    if (args.dataset == 'waterbirds_4_classes_eval' ):
        dir = "./data"
        file_name = "shortcut_candidates_"+args.log_dir.split("/")[-1] + ".csv"
        #dir_test = os.path.join(dir, "test"+file_name)
        dir = os.path.join(dir,file_name)
        # hie das einmal in debug testen:         pd.DataFrame(shortcut_candidates).to_csv(dir_test)
        with open(dir, "w") as f:
            f.writelines(f"{item}\n" for item in shortcut_candidates)
        f.close()
        file_name = "shortcut_candidates_remove_" + args.log_dir.split("/")[-1] + ".csv"
        #dir_test = os.path.join(dir, "test" + file_name)
        dir = "./data"
        dir = os.path.join(dir, file_name)
        with open(dir, "w") as f:
            f.writelines(f"{item}\n" for item in shortcut_candidates_would_be_removed)
        f.close()

        file_name = "shortcut_counter_gr_5_" + args.log_dir.split("/")[-1] + ".csv"
        dir = "./data"
        dir = os.path.join(dir, file_name)
        with open(dir, "w") as f:
            f.writelines(f"{item}\n" for item in shortcuts_counter_gr_5)
        f.close()

        #pd.DataFrame(shortcut_candidates).to_csv(dir_test)


    if (args.dataset == 'isic' ):
        dir = "./data"
        file_name = "shortcut_candidates_"+args.log_dir.split("/")[-1] + ".csv"
        dir_test = os.path.join(dir, "test_"+file_name)
        dir = os.path.join(dir,file_name)
        # hier das einmal in debug testen:         pd.DataFrame(shortcut_candidates).to_csv(dir_test)
        with open(dir, "w") as f:
            f.writelines(f"{item}\n" for item in shortcut_candidates)
        f.close()

        pd.DataFrame(shortcut_candidates).to_csv(dir_test)
        np.save(os.path.join(dir+'shortcut_for_imgs.npy'), shortcut_for_imgs)
    '''
    if (args.dataset == 'isicTEST'):
        net = removeShortcutsManually(net, p)

    with open("./data/ISIC_ALL/shortcut_candidates.txt", "a") as file:
        # Write a line to the file
        for candidate in shortcut_candidates:
            file.write(str(candidate) + "\n")
    print("Abstained: ", abstained, flush=True)'''

    '''
    shortcut_candidates = []
    for p in topks.keys():
        img_iter = tqdm(enumerate(projectloader),
                        total=len(projectloader),
                        mininterval=50.,
                        desc='Visualizing topk',
                        ncols=0)

        shortcut_candidates[p] = False
        for i, (xs, ys) in img_iter:
            img__to_open_name = img_to_open[0].split('/')[-1]
            seg_img_to_open = '/home/thielant/PIPNet/data/segmentations' + img__to_open_name
            seg_img = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(seg_img_to_open))
            seg_img_tensor = transforms.ToTensor()(seg_img).unsqueeze_(0)
            seg_img_tensor_patch = seg_img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
            black_tensor = torch.zeros((3, 32, 32))
            seg_img_tensor_patch_sum = seg_img_tensor_patch.sum()
            if (seg_img_tensor_patch_sum != 0):
                removeShortcutsManually(net, p)
    '''

    all_tensors = []
    for p in []:
        if saved[p]>0:
            # add text next to each topk-grid, to easily see which prototype it is
            text = "P "+str(p)
            txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (0, 0, 0))
            draw = D.Draw(txtimage)
            draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text, anchor='mm', fill="white")
            txttensor = transforms.ToTensor()(txtimage)
            tensors_per_prototype[p].append(txttensor)
            # save top-k image patches in grid
            try:
                grid = torchvision.utils.make_grid(tensors_per_prototype[p], nrow=k+1, padding=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_%s.png"%(str(p))))
                if saved[p]>=k:
                    all_tensors+=tensors_per_prototype[p]
            except:
                pass
    if len(all_tensors)>0:
        grid = torchvision.utils.make_grid(all_tensors, nrow=k+1, padding=1)
        torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_all.png"))
    else:
        print("Pretrained prototypes not visualized. Try to pretrain longer.", flush=True)
    return topks

#@TODO
def visualize_custom_prototype_ids(net, projectloader, num_classes, device, foldername, args: argparse.Namespace, ids):
    print("Visualizing prototypes...", flush=True)
    dir = os.path.join(args.log_dir, foldername,"_custom")
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    abstainedimgs = set()
    notabstainedimgs = set()

    for p in ids:
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p] = near_imgs_dir
        seen_max[p] = 0.
        saved[p] = 0
        saved_ys[p] = []
        tensors_per_prototype[p] = []

    patchsize, skip = get_patch_size(args)

    imgs = projectloader.dataset.imgs

    # skip some images for visualisation to speed up the process
    if len(imgs) / num_classes < 10:
        skip_img = 10
    elif len(imgs) / num_classes < 50:
        skip_img = 5
    else:
        skip_img = 2
    print("Every", skip_img, "is skipped in order to speed up the visualisation process", flush=True)

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=100.,
                    desc='Visualizing custom prototypes',
                    ncols=0)

    # Iterate through the data
    images_seen_before = 0
    for i, (xs, ys) in img_iter:  # shuffle is false so should lead to same order as in imgs
        if ys == 0:
            continue

        if i % skip_img == 0:
            images_seen_before += xs.shape[0]
            continue

        xs, ys = xs.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            softmaxes, _, out = net(xs, inference=True)

        max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
        # In PyTorch, images are represented as [channels, height, width]
        max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
        max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1)
        for p in ids:
            c_weight = torch.max(classification_weights[:, p])  # ignore prototypes that are not relevant to any class
            if c_weight > 0:
                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                w_idx = max_idx_per_prototype_w[p]
                idx_to_select = max_idx_per_prototype[p, h_idx, w_idx].item()
                found_max = max_per_prototype[p, h_idx, w_idx].item()

                imgname = imgs[images_seen_before + idx_to_select]
                if out.max() < 1e-8:
                    abstainedimgs.add(imgname)
                else:
                    notabstainedimgs.add(imgname)


                if found_max > 0:

                    img_to_open = imgs[images_seen_before + idx_to_select]
                    if("/landbird/" in img_to_open[0]):
                        continue
                    if found_max > seen_max[p]:
                        seen_max[p] = found_max

                    if isinstance(img_to_open, tuple) or isinstance(img_to_open,
                                                                    list):  # dataset contains tuples of (img,label)
                        imglabel = img_to_open[1]
                        img_to_open = img_to_open[0]

                    image = transforms.Resize(size=(args.image_size, args.image_size))(
                        Image.open(img_to_open).convert("RGB"))
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0)  # shape (1, 3, h, w)
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size,
                                                                                         softmaxes.shape, patchsize,
                                                                                         skip, h_idx, w_idx)
                    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    saved[p] += 1
                    tensors_per_prototype[p].append((img_tensor_patch, found_max))

                    save_path = os.path.join(dir, "prototype_%s") % str(p)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    draw = D.Draw(image)
                    draw.rectangle([(w_coor_min, h_coor_min), (w_coor_max, h_coor_max)], outline='yellow', width=2)
                    image.save(os.path.join(save_path, 'p%s_%s_%s_%s_rect.png' % (
                    str(p), str(imglabel), str(round(found_max, 2)), str(img_to_open.split('/')[-1].split('.jpg')[0]))))

        images_seen_before += len(ys)

    print("num images abstained: ", len(abstainedimgs), flush=True)
    print("num images not abstained: ", len(notabstainedimgs), flush=True)

    for p in ids:

        saved[p] = 10
        if saved[p] > 0:
            try:
                sorted_by_second = sorted(tensors_per_prototype[p], key=lambda tup: tup[1], reverse=True)
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                torchvision.utils.save_image(grid, os.path.join(dir, "grid_%s.png" % (str(p))))
            except RuntimeError:
                pass

def visualize(net, projectloader, num_classes, device, foldername, args: argparse.Namespace):
    print("Visualizing prototypes...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    abstainedimgs = set()
    notabstainedimgs = set()
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        tensors_per_prototype[p]=[]
    
    patchsize, skip = get_patch_size(args)

    imgs = projectloader.dataset.imgs
    
    # skip some images for visualisation to speed up the process
    if len(imgs)/num_classes <10:
        skip_img=10
    elif len(imgs)/num_classes < 50:
        skip_img=5
    else:
        skip_img = 2
    print("Every", skip_img, "is skipped in order to speed up the visualisation process", flush=True)

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=100.,
                    desc='Visualizing',
                    ncols=0)

    # Iterate through the data
    images_seen_before = 0
    for i, (xs, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if i % skip_img == 0:
            images_seen_before+=xs.shape[0]
            continue
        
        xs, ys = xs.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            softmaxes, _, out = net(xs, inference=True) 

        max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
        # In PyTorch, images are represented as [channels, height, width]
        max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
        max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1)
        for p in range(0, net.module._num_prototypes):
            c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
            if c_weight>0:
                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                w_idx = max_idx_per_prototype_w[p]
                idx_to_select = max_idx_per_prototype[p,h_idx, w_idx].item()
                found_max = max_per_prototype[p,h_idx, w_idx].item()

                imgname = imgs[images_seen_before+idx_to_select]
                if out.max() < 1e-8:
                    abstainedimgs.add(imgname)
                else:
                    notabstainedimgs.add(imgname)
                
                if found_max > seen_max[p]:
                    seen_max[p]=found_max
               
                if found_max > 0.5:
                    img_to_open = imgs[images_seen_before+idx_to_select]
                    if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                        imglabel = img_to_open[1]
                        img_to_open = img_to_open[0]

                    image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open).convert("RGB"))
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)
                    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    saved[p]+=1
                    tensors_per_prototype[p].append((img_tensor_patch, found_max))
                    
                    save_path = os.path.join(dir, "prototype_%s")%str(p)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    draw = D.Draw(image)
                    draw.rectangle([(w_coor_min,h_coor_min), (w_coor_max, h_coor_max)], outline='yellow', width=2)
                    image.save(os.path.join(save_path, 'p%s_%s_%s_%s_rect.png'%(str(p),str(imglabel),str(round(found_max, 2)),str(img_to_open.split('/')[-1].split('.jpg')[0]))))
                    
        
        images_seen_before+=len(ys)

    print("num images abstained: ", len(abstainedimgs), flush=True)
    print("num images not abstained: ", len(notabstainedimgs), flush=True)
    for p in range(net.module._num_prototypes):
        if saved[p]>0:
            try:
                sorted_by_second = sorted(tensors_per_prototype[p], key=lambda tup: tup[1], reverse=True)
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_%s.png"%(str(p))))
            except RuntimeError:
                pass

# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):
    # in case latent output size is 26x26. For convnext with smaller strides. 
    if softmaxes_shape[1] == 26 and softmaxes_shape[2] == 26:
        #Since the outer latent patches have a smaller receptive field, skip size is set to 4 for the first and last patch. 8 for rest.
        h_coor_min = max(0,(h_idx-1)*skip+4)
        if h_idx < softmaxes_shape[-1]-1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize
        w_coor_min = max(0,(w_idx-1)*skip+4)
        if w_idx < softmaxes_shape[-1]-1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
    else:
        h_coor_min = h_idx*skip
        h_coor_max = min(img_size, h_idx*skip+patchsize)
        w_coor_min = w_idx*skip
        w_coor_max = min(img_size, w_idx*skip+patchsize)                                    
    
    if h_idx == softmaxes_shape[1]-1:
        h_coor_max = img_size
    if w_idx == softmaxes_shape[2] -1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size-patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size-patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max
    