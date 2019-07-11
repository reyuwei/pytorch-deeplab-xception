import torch

SMOOTH = 1e-6

class MaskIoUComputation(object):
    def compute(self, logit, target):
        '''
        :param logit: B, C, H, W network prediction mask
        :param target: B, 1, H, W ground truth mask
        :return: iou: B, C
        '''
        B, C, H, W = logit.size()
        logit = torch.max(logit, 1)[1]  # B, H, W
        target = target.squeeze(1)  # B, H, W
        mask_iou = torch.zeros((B, C)).float().cuda()
        trues = torch.ones(logit.size()).int().cuda()
        fals = torch.zeros(logit.size()).int().cuda()

        for i in range(C):
            a = torch.where(logit == i, trues, fals) # B, H, W
            b = torch.where(target == i, trues, fals) # B, H, W

            inter = (a & b).float().sum((1, 2))
            union = (a | b).float().sum((1, 2))

            iou = (inter + SMOOTH) / (union + SMOOTH)
            mask_iou[:, i] = iou

        return mask_iou

if __name__ == "__main__":
    iou = MaskIoUComputation()
    a = torch.rand(4, 11, 513, 513).cuda()
    b = torch.randint(11, (4, 1, 513, 513)).cuda()
    print(iou.compute(a, b))