import torch
import torch.nn as nn

from models.layers import Identity

def initialize_model(config, d_out, is_featurizer=False):
    """
    Initializes models according to the config
        Args:
            - config (dictionary): config dictionary
            - d_out (int): the dimensionality of the model output
            - is_featurizer (bool): whether to return a model or a (featurizer, classifier) pair that constitutes a model.
        Output:
            If is_featurizer=True:
            - featurizer: a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality)
            - classifier: a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear layer.

            If is_featurizer=False:
            - model: a model that is equivalent to nn.Sequential(featurizer, classifier)
    """
    if config.model in ('resnet50', 'resnet34', 'resnet18', 'wideresnet50', 'densenet121'):
        if is_featurizer:
            featurizer = initialize_torchvision_model(
                name=config.model,
                d_out=None,
                **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_torchvision_model(
                name=config.model,
                d_out=d_out,
                **config.model_kwargs)

    elif 'bert' in config.model:
        if is_featurizer:
            featurizer = initialize_bert_based_model(config, d_out, is_featurizer)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_bert_based_model(config, d_out)

    elif config.model == 'resnet18_ms':  # multispectral resnet 18
        from models.resnet_multispectral import ResNet18
        if is_featurizer:
            featurizer = ResNet18(num_classes=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = ResNet18(num_classes=d_out, **config.model_kwargs)

    elif config.model == 'gin-virtual':
        from models.gnn import GINVirtual
        if is_featurizer:
            featurizer = GINVirtual(num_tasks=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = GINVirtual(num_tasks=d_out, **config.model_kwargs)

    elif config.model == 'code-gpt-py':
        from models.code_gpt import GPT2LMHeadLogit, GPT2FeaturizerLMHeadLogit
        from transformers import GPT2Tokenizer
        name = 'microsoft/CodeGPT-small-py'
        tokenizer = GPT2Tokenizer.from_pretrained(name)
        if is_featurizer:
            model = GPT2FeaturizerLMHeadLogit.from_pretrained(name)
            model.resize_token_embeddings(len(tokenizer))
            featurizer = model.transformer
            classifier = model.lm_head
            model = (featurizer, classifier)
        else:
            model = GPT2LMHeadLogit.from_pretrained(name)
            model.resize_token_embeddings(len(tokenizer))

    elif config.model == 'logistic_regression':
        assert not is_featurizer, "Featurizer not supported for logistic regression"
        model = nn.Linear(out_features=d_out, **config.model_kwargs)
    elif config.model == 'unet-seq':
        from models.CNN_genome import UNet
        if is_featurizer:
            featurizer = UNet(num_tasks=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = UNet(num_tasks=d_out, **config.model_kwargs)

    elif config.model == 'fasterrcnn':
        if is_featurizer: # TODO
            raise NotImplementedError('Featurizer not implemented for detection yet')
        else:
            model = initialize_fasterrcnn_model(config, d_out)
        model.needs_y = True

    else:
        raise ValueError(f'Model: {config.model} not recognized.')

    # The `needs_y` attribute specifies whether the model's forward function
    # needs to take in both (x, y).
    # If False, Algorithm.process_batch will call model(x).
    # If True, Algorithm.process_batch() will call model(x, y) during training,
    # and model(x, None) during eval.
    if not hasattr(model, 'needs_y'):
        # Sometimes model is a tuple of (featurizer, classifier)
        if isinstance(model, tuple):
            for submodel in model:
                submodel.needs_y = False
        else:
            model.needs_y = False

    return model


def initialize_bert_based_model(config, d_out, is_featurizer=False):
    from models.bert.bert import BertClassifier, BertFeaturizer
    from models.bert.distilbert import DistilBertClassifier, DistilBertFeaturizer

    if config.model == 'bert-base-uncased':
        if is_featurizer:
            model = BertFeaturizer.from_pretrained(config.model, **config.model_kwargs)
        else:
            model = BertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)
    elif config.model == 'distilbert-base-uncased':
        if is_featurizer:
            model = DistilBertFeaturizer.from_pretrained(config.model, **config.model_kwargs)
        else:
            model = DistilBertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)
    else:
        raise ValueError(f'Model: {config.model} not recognized.')
    return model

def initialize_torchvision_model(name, d_out, **kwargs):
    import torchvision

    # get constructor and last layer names
    if name == 'wideresnet50':
        constructor_name = 'wide_resnet50_2'
        last_layer_name = 'fc'
    elif name == 'densenet121':
        constructor_name = name
        last_layer_name = 'classifier'
    elif name in ('resnet50', 'resnet34', 'resnet18'):
        constructor_name = name
        last_layer_name = 'fc'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else: # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)
    return model


def initialize_fasterrcnn_model(config, d_out):

    from models.detection.fasterrcnn import fasterrcnn_resnet50_fpn

    # load a model pre-trained pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(
        pretrained=config.model_kwargs["pretrained_model"],
        pretrained_backbone=config.model_kwargs["pretrained_backbone"],
        num_classes=d_out,
        min_size=config.model_kwargs["min_size"],
        max_size=config.model_kwargs["max_size"]
        )

    return model
