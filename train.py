from argparse import ArgumentParser
from src.model import SimCLR
import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    stl10_normalization,
)
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    print("START PROGRAM")

    
    #######################
    # PARSE ARGUMENTS
    #######################
    print("PARSING ARGUMENTS...")
    parser = ArgumentParser()

    

    # add model specific args
    parser = SimCLR.add_model_specific_args(parser)

    args = parser.parse_args()

    ###########################
    # INITIALIZE DATAMODULE
    ###########################
    print("INITIALIZING DATAMODULE...")
    if args.dataset == "stl10":
        dm = STL10DataModule(data_dir='./data', batch_size=args.batch_size, num_workers=args.num_workers)
        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples
        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm.size()[-1]
        normalization = stl10_normalization()
        args.gaussian_blur = True
        args.jitter_strength = 1.0

    elif args.dataset == "cifar10":
        val_split = 5000
        if args.num_nodes * args.gpus * args.batch_size > val_split:
            val_split = args.num_nodes * args.gpus * args.batch_size
        dm = CIFAR10DataModule(
            data_dir='./data', batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split
        )
        args.num_samples = dm.num_samples
        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]
        args.temperature = 0.5
        normalization = cifar10_normalization()
        args.gaussian_blur = False
        args.jitter_strength = 0.5


    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    
    
    ###########################
    # INITIALIZE MODEL
    ###########################
    print("INITIALIZING MODEL...")
    model = SimCLR(**args.__dict__)



    ###########################
    # INITIALIZE LOGGER
    ###########################
    print("INITIALIZING LOGGER...")
    logdir = 'logs'
    logdir += datetime.now().strftime("/%m%d")
    logdir += '/{}'.format(args.dataset)
    logdir += '/{}epochs'.format(args.max_epochs)
    logdir += '/{}'.format(args.optimizer)
    logger = TensorBoardLogger(logdir, name='')

    ###########################
    # TRAIN
    ###########################
    print("START TRAINING...")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss")
    callbacks = [model_checkpoint]
    callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        accelerator="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        logger = logger
    )

    trainer.fit(model, datamodule=dm)


    ###########################
    # TEST
    ###########################
    print("START TESTING...")
    trainer.save_checkpoint("{0}.ckpt".format(datetime.now().strftime("/%m%d")))
    # Some testing code

if __name__=='__main__':
    main()