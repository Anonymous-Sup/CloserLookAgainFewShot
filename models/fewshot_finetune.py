"""
module for gradient-based test-time methods, e.g., finetune, eTT, TSA, URL, Cosine Classifier
"""
from architectures import get_backbone, get_classifier
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
import torch

def create_label_map(support_labels, query_labels):
    """
    Create a consistent label mapping for both support and query labels.
    This ensures that the same original labels in support and query are mapped to the same new labels.
    """
    # Combine unique labels from both support and query sets
    unique_labels = torch.unique(torch.cat([support_labels, query_labels]))
    # Create a mapping from the original labels to a contiguous range starting from 0
    label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
    return label_map

def relabel(labels, label_map):
    """
    Apply the label_map to relabel the given labels.
    """
    relabeled = torch.tensor([label_map[label.item()] for label in labels], dtype=torch.int64)
    return relabeled

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

        for i, img_task in enumerate(zip(support_imgs, query_imgs, support_labels, query_labels)):
            support_img, query_img, s_labels, q_labels = img_task

            # Create a consistent label map for both support and query labels
            label_map = create_label_map(s_labels.squeeze_().cuda(), q_labels.squeeze_().cuda())

            # Relabel support and query labels using the same mapping
            s_labels = relabel(s_labels.squeeze_().cuda(), label_map)
            q_labels = relabel(q_labels.squeeze_().cuda(), label_map)

            # Pass relabeled support labels to the classifier along with the images
            score = self.classifier(query_img.squeeze_().cuda(), support_img.squeeze_().cuda(), s_labels, **kwargs)

            # Compute loss and accuracy using the relabeled query labels
            loss += F.cross_entropy(score, q_labels)
            acc.append(accuracy(score, q_labels)[0])

        loss = loss / len(support_imgs)
        return loss, acc
    
    # def test_forward(self, support_imgs, query_imgs, support_labels, query_labels, *args, **kwargs):
    #     loss = 0.
    #     acc = []
    #     for i , img_task in enumerate(zip(support_imgs, query_imgs, support_labels, query_labels)):
    #         support_img, query_img, _, _ = img_task
    #         print("queruy_img",query_img.shape)
    #         print("support_img",support_img.shape)
    #         print("support_labels",support_labels[i].shape)
    #         print("query_labels",query_labels[i].shape)
    #         score = self.classifier(query_img.squeeze_().cuda(), support_img.squeeze_().cuda(), support_labels[i].squeeze_().cuda(), **kwargs)
    #         loss += F.cross_entropy(score, query_labels[i].squeeze_().cuda())
    #         acc.append(accuracy(score, query_labels[i].squeeze_().cuda())[0])
    #     loss = loss / len(support_imgs)
    #     return loss, acc
    

def get_model(config):
    return FinetuneModule(config)