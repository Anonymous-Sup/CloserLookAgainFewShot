"""
module for gradient-based test-time methods, e.g., finetune, eTT, TSA, URL, Cosine Classifier
"""
from architectures import get_backbone, get_classifier
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy

class FinetuneModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)

        # The last hyperparameter is the head mode
        self.mode = config.MODEL.CLASSIFIER_PARAMETERS[-1]

        if not self.mode == "NCC":
            classifier_hyperparameters = [self.backbone]+config.MODEL.CLASSIFIER_PARAMETERS
            self.classifier = get_classifier(config.MODEL.CLASSIFIER, *classifier_hyperparameters)
    
    def append_adapter(self):
        # append adapter to the backbone
        self.backbone = get_backbone("resnet_tsa",backbone=self.backbone)
        classifier_hyperparameters = [self.backbone]+self.config.MODEL.CLASSIFIER_PARAMETERS
        self.classifier = get_classifier(self.config.MODEL.CLASSIFIER, *classifier_hyperparameters)

    # def test_forward(self, img_tasks, label_tasks, *args, **kwargs):
    #     batch_size = len(img_tasks)
    #     loss = 0.
    #     acc = []
    #     for i, img_task in enumerate(img_tasks):
    #         score = self.classifier(img_task["query"].squeeze_().cuda(), img_task["support"].squeeze_().cuda(),
    #                                 label_tasks[i]["support"].squeeze_().cuda())
    #         loss += F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())
    #         acc.append(accuracy(score, label_tasks[i]["query"].cuda())[0])
    #     loss /= batch_size
    #     return loss, acc
    
    def test_forward(self, support_imgs, query_imgs, support_labels, query_labels, *args, **kwargs):
        loss = 0.
        acc = []
        for i , img_task in enumerate(zip(support_imgs, query_imgs, support_labels, query_labels)):
            support_img, query_img, _, _ = img_task
            print("queruy_img",query_img.shape)
            print("support_img",support_img.shape)
            print("support_labels",support_labels[i].shape)
            print("query_labels",query_labels[i].shape)
            score = self.classifier(query_img.squeeze_().cuda(), support_img.squeeze_().cuda(), support_labels[i].squeeze_().cuda(), **kwargs)
            loss += F.cross_entropy(score, query_labels[i].squeeze_().cuda())
            acc.append(accuracy(score, query_labels[i].squeeze_().cuda())[0])
        loss = loss / len(support_imgs)
        return loss, acc
    

def get_model(config):
    return FinetuneModule(config)