import os
import shutil
import torch
from collections import OrderedDict
import glob
import cv2
import numpy as np
from dataloaders.utils import decode_segmap

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_demo_result(self, pred_dict, image_list, root_folder):
        predictions = pred_dict['images']
        ious = pred_dict['ious']
        all_idx = np.arange(11)

        imgfolder = os.path.join(root_folder, "mask//mask_refine_network")
        initmaskfolder = os.path.join(root_folder, "mask")
        if not os.path.exists(imgfolder):
            os.mkdir(imgfolder)
        for predi in range(len(predictions)):
            pred_labels = np.unique(predictions[predi])
            iou = ious[predi]
            iou[~np.isin(all_idx, pred_labels)] = -1
            initmask = cv2.imread(os.path.join(initmaskfolder, image_list[predi]))
            h, w, c = initmask.shape
            imgname = os.path.join(imgfolder, image_list[predi])
            segmap, segmap_gray = decode_segmap(predictions[predi], dataset='semantic_body')
            segmap_gray = cv2.resize(segmap_gray, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(imgname, segmap_gray)
            scorename, _ = os.path.splitext(imgname)
            np.savetxt(scorename + ".txt", iou[1:], fmt="%.6f")

    def save_images(self, pred_dict):
        predictions = pred_dict['images']
        ious = pred_dict['ious']
        targets = pred_dict['targets']
        inputs = pred_dict['inputs']
        assert(len(predictions) == len(ious))
        imgfolder = self.directory + "//test/"
        if not os.path.exists(imgfolder):
            os.mkdir(imgfolder)
        for predi in range(len(predictions)):
            str_iou = "{:.05f}".format(ious[predi])
            imgname = os.path.join(imgfolder, str(predi) + "_" + str(str_iou) + ".png")
            segmap = decode_segmap(predictions[predi], dataset='semantic_body')
            segmap_target = decode_segmap(targets[predi], dataset='semantic_body')

            img_tmp = np.transpose(inputs[predi], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)

            save = np.zeros((segmap.shape[0], segmap.shape[1]*3, 3), dtype=np.uint8)
            save[0:segmap.shape[0], 0:segmap.shape[1], :] = segmap
            save[0:segmap.shape[0], segmap.shape[1]:segmap.shape[1]*2, :] = segmap_target
            save[0:segmap.shape[0], segmap.shape[1]*2:, :] = img_tmp

            cv2.imwrite(imgname, save)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))



    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
