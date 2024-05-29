import numpy as np


class SegmentationEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def F_Score(self):
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP 
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F = 2 * P * R / (P + R)
        return P, R, F

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        IoUs = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return IoUs

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        
    def print_result(self, task):
        P, R, F = self.F_Score()
        IoUs = self.Mean_Intersection_over_Union()
        
        if task == 'text segmentation':
            print(f'fgIoU: {IoUs[1]}; P: {P[1]}; R: {R[1]}; F: {F[1]}')
        elif task == 'tampered text detection':
            print('Real Text:')
            print(f'  IoU: {IoUs[1]}; P: {P[1]}; R: {R[1]}; F: {F[1]}')
            print('Tampered Text:')
            print(f'  IoU: {IoUs[2]}; P: {P[2]}; R: {R[2]}; F: {F[2]}')
            print('Average:')
            print(f'  mIoU: {np.nanmean(IoUs[-2:])}; mF: {np.nanmean(F[-2:])}')
        else:
            raise ValueError
        