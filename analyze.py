from argparse import ArgumentParser
import pytorch_lightning as pl
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from pl_bolts.datamodules.stl10_datamodule import STL10DataModule
from pl_bolts.models.self_supervised import SimCLR, CPC_v2, SwAV
from pl_bolts.models.self_supervised.resnets import resnet50

import torch
import torch.nn as nn
import numpy
import os
import datetime
from src.feature_tsne import featureToTSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    print("START PROGRAM")

    ###########################
    # PARSE ARGUMENTS
    ###########################
    print("PARSING ARGUMENTS...")
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--model", default="SimCLR", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--pca", default=0, type=int)
    
    args = parser.parse_args()
    

    ###########################
    # INITIALIZE DATAMODULE
    ###########################
    print("INITIALIZING DATAMODULE...")

    if not os.path.exists(os.path.dirname('./data')):
        os.makedirs(os.path.dirname('./data'))
    if args.dataset=="stl10":
        STL10DataModule().prepare_data()
        dm = STL10DataModule(data_dir="data", batch_size=args.batch_size)
        dm.prepare_data()
        dm.setup()
        dm.train_dataloader = dm.train_dataloader_labeled
        dm.val_dataloader = dm.val_dataloader_labeled
    elif args.dataset=="cifar10":
        CIFAR10DataModule().prepare_data()
        dm = CIFAR10DataModule(data_dir="data", batch_size=args.batch_size, num_workers=8)
        dm.prepare_data()
        dm.setup()
    

    ###########################
    # SET DEVICE
    ###########################
    print("SETTING GPU DEVICE...")
    device = torch.device('cuda:0')
    

    ###########################
    # LOAD PRETRAINED MODEL
    ###########################
    print("LOADING PRETRAINED MODEL...")
    model_encoder = resnet50()

    if args.model=="SimCLR":
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        model = SimCLR.load_from_checkpoint(weight_path, strict=False)
        model_encoder.load_state_dict(model.encoder.state_dict(), strict=False)
        
    elif args.model=="SwAV":
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/bolts_swav_imagenet/swav_imagenet.ckpt'
        model = SwAV.load_from_checkpoint(weight_path, strict=False)
        model_encoder.load_state_dict(model.model.state_dict(), strict=False)
        
    elif args.model=="MocoV2":
        weight_path = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
        state_dict = torch.hub.load_state_dict_from_url(weight_path, progress=False)['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model_encoder.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        
    elif args.model=="SimSiam":
        weight_path = "https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar"
        state_dict = torch.hub.load_state_dict_from_url(weight_path, progress=False)['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder'):
                # remove prefix
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model_encoder.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        
    elif args.model=="Supervised":
        model_encoder.load_state_dict(resnet50(pretrained=True).state_dict(), strict=False)

    elif args.model=="Untrained":
        pass;
    
    model_encoder.avgpool = nn.Identity()
    model_encoder.fc = nn.Identity()        
    model_encoder.to(device)
    model_encoder.eval()

    ###########################
    # ENCODING TEST DATA
    ###########################
    print("ENCODING TEST DATA...")
    test_data_encoded = torch.zeros([len(dm.dataset_test), 2048]).to(device)
    test_data_labels = torch.zeros([len(dm.dataset_test)])
    for xi, x in enumerate(dm.test_dataloader()):
        input, label = x
        encoded = model_encoder(input.to(device))
        test_data_encoded[xi*args.batch_size:min((xi+1)*args.batch_size, len(dm.dataset_test))] = encoded[0]
        test_data_labels[xi*args.batch_size:min((xi+1)*args.batch_size, len(dm.dataset_test))] = label
    features = test_data_encoded.cpu().detach().numpy()
    

    ##################################
    # APPLY PCA
    ##################################
    if args.pca>0:
        print("APPLYING PCA TO FEATURE VECTOR...")
        scaler = StandardScaler()    
        scaler.fit(features)
        features_scaled = scaler.transform(features)
        pca = PCA(n_components=args.pca)
        pca.fit(features_scaled)
        features = pca.transform(features_scaled)


    ##################################
    # VISUALIZING ENCODING VECTORS
    ##################################
    print("VISUALIZING ENCODING VECTORS...")

    directory = "./tsne_results/{0}/{1}/".format(args.dataset, args.model)
    if not os.path.exists(os.path.dirname(directory)):
        os.makedirs(os.path.dirname(directory))
    if args.pca>0:
        featureToTSNE(features, test_data_labels.type(torch.int32), "{0}/pca{1}_{2}.png".format(directory, args.pca, datetime.datetime.now().strftime("%m%d%H%M%S")))
    else:
        featureToTSNE(features, test_data_labels.type(torch.int32), "{0}/{1}.png".format(directory, datetime.datetime.now().strftime("%m%d%H%M%S")))

    
if __name__=='__main__':
    main()


