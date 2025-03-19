from PIL import Image
from torchvision import transforms
from util import args
from util.func import get_patch_size
from util.vis_pipnet import get_img_coordinates
import torch



#def checkShortcuts(img_name, patchsize, min_height, min_width, seg_path):
def checkShortcuts(net,xs,prototype_idx,max_h, max_idx_h,max_w, max_idx_w):
    seg_path = "/home/thielant/PIPNet/data/segmentations/001.Black_footed_Albatross/Black_Footed_Albatross_0049_796063"

    patchsize, skip = get_patch_size(args)
    softmaxes, pooled, out = net(xs,
                                 inference=True)  # softmaxes has shape (bs, num_prototypes, W, H), pooled has shape (bs, num_prototypes), out has shape (bs, num_classes)

    max_idx_h = max_idx_h[max_idx_w].item()
    max_idx_w = max_idx_w.item()
    image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(seg_path))
    img_tensor = transforms.ToTensor()(image).unsqueeze_(0)  # shape (1, 3, h, w)
    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize,
                                                                         skip, max_idx_h, max_idx_w)
    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
    img_patch = transforms.ToPILImage()(img_tensor_patch)
    img_patch.save("/home/thielant/PIPNet/data/segmentations/patch_test.jpg")
