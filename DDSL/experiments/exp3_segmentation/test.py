import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from train import masked_miou, AverageMeter, compose_masked_img, validate
from loader import CityScapeLoader
from torch.utils.data import Dataset, DataLoader
from model import BatchedRasterLoss2D, SmoothnessLoss, PolygonNet
import importlib.machinery

import argparse, time, os
from tqdm import tqdm
from tabulate import tabulate


def evaluate(args, model, loader, criterion, criterion_smooth, epoch, device):
    model.eval()

    rasterlosses = [AverageMeter() for _ in range(len(args.res))]
    smoothloss = AverageMeter()
    losses_sum = AverageMeter()
    mious = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    count = 0

    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, target, label) in enumerate(tqdm(loader)):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            num = input.size(0)
            input, label = input.to(device), label.to(device)
            target = [t.to(device) for t in target]
            output = model(input) # output shape [N, 128, 2]

            evals = [crit(output[:, ::l], t) for crit, l, t in zip(criterion, args.levels, target)]
            loss_vecs = [e[0] for e in evals]
            output_rasters = [e[1] for e in evals]
            rasterlosses_ = [(lv * args.weights[label]).sum() for lv in loss_vecs]

            rasterloss = torch.sum(torch.stack(rasterlosses_, dim=0), dim=0)
            output_raster = output_rasters[0]

            # compute smoothness loss
            smoothloss_ = criterion_smooth(output)
            smoothloss_ = (smoothloss_).mean()

            loss = rasterloss + args.smooth_loss * smoothloss_

            # measure miou and record loss
            miou, miou_count = masked_miou(output_raster, target[0], label, args.nclass)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # update statistics
            for i in range(len(args.res)):
                rasterlosses[i].update(rasterlosses_[i].item())
            smoothloss.update(smoothloss_)
            mious.update(miou, miou_count)
            losses_sum.update(rasterloss.item(), num)

            # output visualization
            if count < args.nsamples:
                # create image grid
                fname = os.path.join(args.output_dir, "sample_{}.png".format(count))
                compimg = compose_masked_img(output_raster, target[0], input, args.img_mean, args.img_std)
                imgrid = vutils.save_image(compimg, fname, normalize=False, scale_each=False)

            count += 1

        log_text = ('Test Epoch: [{0}]\t'
                    'CompTime {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
                    'DataTime {data_time.sum:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.avg:.4f}\t'
                    'mIoU {miou:.3f}\t').format(
                     epoch, batch_time=batch_time,
                     data_time=data_time, loss=losses_sum, miou=mious.avgcavg)
        print(log_text)
        # tabulate mean iou 
        print(tabulate(dict(zip(args.label_names, [[iou] for iou in mious.avg])), headers="keys"))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for test (default: 64)')
    parser.add_argument('--loss_type', type=str, choices=['l1', 'l2'], default='l1', help='type of loss for raster loss computation')
    parser.add_argument('--nlevels', type=int, default=4, help="number of polygon levels, higher->finer")
    parser.add_argument('--feat', type=int, default=256, help="number of base feature layers")
    parser.add_argument('--dropout', action='store_true', help="dropout during training")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data_folder', type=str, default="mres_processed_data",
                        help='path to data folder (default: mres_processed_data)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/checkpoint_polygonnet_best.pth.tar', help="path to checkpoint to load")
    parser.add_argument('--output_dir', type=str, default='output_vis_ins', help="directory to output visualizations")
    parser.add_argument('--nsamples', type=int, default=0, help='number of samples to produce.')
    parser.add_argument('--multires', action='store_true', help="multiresolution loss")
    parser.add_argument('--transpose', action='store_true', help="transpose target")
    parser.add_argument('--smooth_loss', default=1.0, type=float, help="smoothness loss multiplier (0 for none)")
    parser.add_argument('--uniform_loss', action='store_true', help="use same loss regardless of category frequency")
    parser.add_argument('--workers', default=12, type=int, help="number of data loading workers")

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    args.label_names = ["car", "truck", "train", "bus", "motorcycle", "bicycle", "rider", "person"]
    args.instances = torch.tensor([30246, 516, 76, 325, 658, 3307, 1872, 19563]).double().to(device)
    args.nclass = len(args.label_names)
    args.weights = 1 / torch.log(args.instances / args.instances.sum() + 1.02)
    args.weights /= args.weights.sum()

    if not os.path.exists(args.output_dir) and args.nsamples > 0:
        os.makedirs(args.output_dir)

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    torch.manual_seed(args.seed)

    # get training / valid sets
    args.img_mean = [0.485, 0.456, 0.406]
    args.img_std = [0.229, 0.224, 0.225]
    normalize = torchvision.transforms.Normalize(mean=args.img_mean,
                                                  std=args.img_std)
    # DO NOT include horizontal and vertical flips in the composed transforms!
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    testset = CityScapeLoader(args.data_folder, "test", transforms=transform, RandomHorizontalFlip=0.0, RandomVerticalFlip=0.0, mres=args.multires, transpose=args.transpose)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, drop_last=False, num_workers=args.workers, pin_memory=True)
    # initialize and parallelize model
    model = PolygonNet(nlevels=args.nlevels, dropout=args.dropout, feat=args.feat)
    model = nn.DataParallel(model)
    model.to(device)

    # Multi-Resolution
    n = model.module.npoints
    if args.multires:
        args.res = [224, 112, 56, 28]
        args.npt = [n, int(n/2), int(n/4), int(n/8)]
        args.levels = [1, 2, 4, 8]
    else:
        args.res = [224]
        args.npt = [n]
        args.levels = [1]

    if os.path.isfile(args.ckpt):
        print("=> loading checkpoint '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt)
        args.best_miou = checkpoint['best_miou']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.ckpt))

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    rres = [True, False, False, False] if args.multires else [True]
    criterion = [BatchedRasterLoss2D(npoints=n, res=r, loss=args.loss_type, return_raster=rr).to(device) for n, r, rr in zip(args.npt, args.res, rres)]
    criterion_smooth = SmoothnessLoss()
    evaluate(args, model, test_loader, criterion, criterion_smooth, checkpoint['epoch'], device)

if __name__ == '__main__':
    main()
