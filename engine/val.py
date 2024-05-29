import os 
import cv2
import torch 
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm 
from utils.visualize import tensor_to_cv2image
from eval.segmentation import SegmentationEvaluator


@torch.no_grad()
def evaluate(model, data_loader, args):
    model.eval()
    EVALUATE_FUNC[args.eval_data_cfg['TYPE']](model, data_loader, args)


def evaluate_textremoval(model, data_loader, args):
    device = torch.device(args.device)

    save_folder = os.path.join(args.output_dir, 'SCUT-EnsText')
    os.makedirs(save_folder, exist_ok=True)

    for data in tqdm(data_loader):
        images = data['image'].to(device)
        tasks = data['task']

        outputs = model(images, tasks=tasks)[-1]
        outputs = torch.clamp(outputs, min=0, max=1)
        for i, output in enumerate(outputs):
            image_path = data['filepath'][i]
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(save_folder, image_name + '.png')
            output = tensor_to_cv2image(output.cpu(), False)
            cv2.imwrite(save_path, output)


def evaluate_textseg(model, data_loader, args):
    device = torch.device(args.device)
    evaluator = SegmentationEvaluator(2)

    for data in tqdm(data_loader):
        images = data['image'].to(device)
        tasks = data['task']

        outputs = model(images, tasks=tasks)[-1]
        outputs = torch.clamp(outputs, min=0, max=1)
    
        labels = data['label']
        filepaths = data['filepath']
        for output, label, filepath in zip(outputs, labels, filepaths):
            h, w = label.shape
            output = F.interpolate(output.unsqueeze(0), (h, w), mode='bilinear', align_corners=False)[0]
            output = output.cpu()
            if args.visualize:
                seg_map = tensor_to_cv2image(output, False)
                save_folder = os.path.join(args.output_dir, 'TextSeg')
                os.makedirs(save_folder, exist_ok=True)
                save_path = os.path.join(save_folder, os.path.basename(filepath))
                cv2.imwrite(save_path, seg_map)

            output = output.numpy()
            output = output.mean(0)
            bi_output = np.zeros(output.shape).astype('int')
            bi_output[output > args.textseg_conf_thres] = 1
            label[label==255] = 1
            evaluator.add_batch(label, bi_output)

    evaluator.print_result(task='text segmentation')
        

def evaluate_ttd(model, data_loader, args):
    device = torch.device(args.device)
    evaluator = SegmentationEvaluator(3)

    for data in tqdm(data_loader):
        images = data['image'].to(device)
        tasks = data['task']

        outputs = model(images, tasks=tasks)
        outputs = torch.clamp(outputs[-1], min=0, max=1)
    
        labels = data['label']
        filepaths = data['filepath']
        for output, label, filepath in zip(outputs, labels, filepaths):
            h, w = label.shape[:2]
            output = F.interpolate(output.unsqueeze(0), (h, w), mode='bilinear', align_corners=False)[0]

            output = output.cpu()
            if args.visualize:
                seg_map = tensor_to_cv2image(output, False)
                seg_map = seg_map[:, :, ::-1] # keep colors for three categories consistent with visualizations in the paper
                save_folder = os.path.join(args.output_dir, 'TTD')
                os.makedirs(save_folder, exist_ok=True)
                save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(filepath))[0] + '.png')
                cv2.imwrite(save_path, seg_map)

            output = output.numpy()
            output = output.argmax(axis=0)
            label = label.argmax(axis=-1)
            evaluator.add_batch(label, output)

    evaluator.print_result(task='tampered text detection')


EVALUATE_FUNC = {
    'text removal': evaluate_textremoval,
    'text segmentation': evaluate_textseg,
    'tampered text detection': evaluate_ttd
}