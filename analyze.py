from argparse import ArgumentParser
import pytorch_lightning as pl
from pl_bolts.datamodules.cifar10_datamodule import CIFAR10DataModule
from pl_bolts.datamodules.stl10_datamodule import STL10DataModule
from pl_bolts.models.self_supervised import SimCLR, CPC_v2, SwAV
from pl_bolts.models.self_supervised.resnets import resnet50
from torchsummary import summary
from tabulate import tabulate


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
    parser.add_argument("--pretrain_dataset", default="imagenet", type=str)
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
    if args.pretrain_dataset=="imagenet":
        model_encoder = resnet50()
        fv_size = 2048
    elif args.pretrain_dataset=="cifar10":
        model_encoder = resnet50(first_conv= False, maxpool1=False)
        fv_size = 32768
    
    if args.model=="SimCLR":
        if args.pretrain_dataset=="imagenet":
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        elif args.pretrain_dataset=="cifar10":
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-sgd/simclr-cifar10-sgd.ckpt'
        model = SimCLR.load_from_checkpoint(weight_path, strict=False)
        model_encoder.load_state_dict(model.encoder.state_dict(), strict=False)
        
        
    elif args.model=="SwAV":
        if args.pretrain_dataset=="imagenet":
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/bolts_swav_imagenet/swav_imagenet.ckpt'
        model = SwAV.load_from_checkpoint(weight_path, strict=False)
        model_encoder.load_state_dict(model.model.state_dict(), strict=False)
        
    elif args.model=="MocoV2":
        if args.pretrain_dataset=="imagenet":
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
        if args.pretrain_dataset=="imagenet":
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
        if args.pretrain_dataset=="imagenet":
            model_encoder.load_state_dict(resnet50(pretrained=True).state_dict(), strict=False)
        elif args.pretrain_dataset=="cifar10":
            pass
            # state_dict = torch.load('./pretrained_weights/resnet50_cifar10_ce.pt')
            # model_encoder.load_state_dict(state_dict, strict=True)

    elif args.model=="Untrained":
        pass
    
    model_encoder.avgpool = nn.Identity()
    model_encoder.fc = nn.Identity()        
    model_encoder.to(device)
    model_encoder.eval()

    ###########################
    # ENCODING TEST DATA
    ###########################
    print("ENCODING TEST DATA...")
    
    test_data_encoded = torch.zeros([len(dm.dataset_test), fv_size]).to(device)
    test_data_labels = torch.zeros([len(dm.dataset_test)])
    with torch.no_grad():
        for xi, x in enumerate(dm.test_dataloader()):
            input, label = x
            # print(input.shape)
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
        pca_features = pca.transform(features_scaled)        

    ######################################
    # COMPARE MEAN VECTORS & EIGENVALUES
    ######################################
    print("COMPARING EIGENVALUES AND MEAN REPRESENTATIONS...")
    cos = nn.CosineSimilarity(dim=0, eps=1e-8)
    mean_features = torch.zeros([10, fv_size])
    for i in range(10):
        feature_indices = ((test_data_labels==i).nonzero(as_tuple=True)[0])
        mean_feature = numpy.mean(features[feature_indices], axis=0)
        mean_features[i] = torch.from_numpy(mean_feature)

    if args.pca>0:
        eigenvalues = numpy.array(pca.components_)
        table_data = []
        for i in range(args.pca):
            row_data = []
            for j in range(10):
                row_data.append(cos(torch.from_numpy(eigenvalues[i]), mean_features[j]).item())
            table_data.append(row_data)
        print(tabulate(table_data))

    print("COMPARING MEAN REPRESENTATIONS...")
    table_data = []
    for i in range(10):
        row_data = []
        for j in range(10):
            row_data.append(cos(mean_features[i], mean_features[j]).item())
        table_data.append(row_data)
    print(tabulate(table_data))

    ##################################
    # VISUALIZING ENCODING VECTORS
    ##################################
    print("VISUALIZING ENCODING VECTORS...")
    directory = "./tsne_results/{0}/{1}/pretrained_{2}/".format(args.dataset, args.model, args.pretrain_dataset)
    if not os.path.exists(os.path.dirname(directory)):
        os.makedirs(os.path.dirname(directory))
    if args.pca>0:
        featureToTSNE(features, test_data_labels.type(torch.int32), "{0}/pca{1}_{2}.png".format(directory, args.pca, datetime.datetime.now().strftime("%m%d%H%M%S")))
    else:
        featureToTSNE(features, test_data_labels.type(torch.int32), "{0}/{1}.png".format(directory, datetime.datetime.now().strftime("%m%d%H%M%S")))


    
        
    


    
if __name__=='__main__':
    main()


