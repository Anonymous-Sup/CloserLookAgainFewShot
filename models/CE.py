"""
Simple supervised Cross Entropy (CE) training used widely in image classification.
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

class CrossEntropyTraining(nn.Module):
    def __init__(self, config, num_classes):
        """
        config: configuration file
        num_classes: a list containing number of classes per dataset. 
                    (For CE, the number of datasets must be 1)
        """
        super().__init__()
        # assert len(num_classes) == 1
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)
        self.classifier = nn.Linear(self.backbone.outdim, num_classes)
        self.val_test_classifier = get_classifier(config.MODEL.CLASSIFIER, *config.MODEL.CLASSIFIER_PARAMETERS)
    

    def train_forward(self, imgs, labels, *args, **kwargs):
        imgs = imgs.squeeze_().cuda()
        labels = labels.squeeze_().cuda()

        features = self.backbone(imgs)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).squeeze_(-1).squeeze_(-1)
        score = self.classifier(features)
        loss = F.cross_entropy(score, labels)
        acc = [accuracy(score, labels)[0]]
        return loss, acc

    def val_test_forward(self, support_imgs, query_imgs, support_labels, query_labels, *args, **kwargs):
        loss = 0.
        acc = []
        for i , img_task in enumerate(zip(support_imgs, query_imgs, support_labels, query_labels)):
            support_img, query_img, s_labels, q_labels = img_task

            # Create a consistent label map for both support and query labels
            label_map = create_label_map(s_labels.squeeze_().cuda(), q_labels.squeeze_().cuda())

            # Relabel support and query labels using the same mapping
            s_labels = relabel(s_labels.squeeze_().cuda(), label_map)
            q_labels = relabel(q_labels.squeeze_().cuda(), label_map)
            
            
            support_features = self.backbone(support_img.squeeze_().cuda())
            query_features = self.backbone(query_img.squeeze_().cuda())

            if support_features.dim() == 4:
                support_features = F.adaptive_avg_pool2d(support_features, 1).squeeze_(-1).squeeze_(-1)
                query_features = F.adaptive_avg_pool2d(query_features, 1).squeeze_(-1).squeeze_(-1)
            
            score = self.val_test_classifier(query_features, support_features, s_labels.cuda(), **kwargs)
            
            loss += F.cross_entropy(score, q_labels.cuda())
            
            acc.append(accuracy(score, q_labels.cuda())[0])
        # for i, img_task in enumerate(img_tasks):
        #     support_features = self.backbone(img_task["support"].squeeze_().cuda())
            
        #     query_features = self.backbone(img_task["query"].squeeze_().cuda())
            
        #     score = self.val_test_classifier(query_features, support_features,
        #                             label_tasks[i]["support"].squeeze_().cuda(), **kwargs)
            
        #     loss += F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())
        #     acc.append(accuracy(score, label_tasks[i]["query"].cuda())[0])
        return loss, acc
    
    def val_forward(self, support_imgs, query_imgs, support_labels, query_labels, *args, **kwargs):
        # loss, acc = model.test_forward(imgs, labels, dataset_index)
        return self.val_test_forward(support_imgs, query_imgs, support_labels, query_labels)
    
    def test_forward(self, support_imgs, query_imgs, support_labels, query_labels, *args,  **kwargs):
        return self.val_test_forward(support_imgs, query_imgs, support_labels, query_labels)

def get_model(config, num_classes):
    return CrossEntropyTraining(config, num_classes)