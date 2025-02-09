'''
this project were based the pytorch, pytorch lightning and pytorch video library, 
for rapid development.
'''

# %%
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
# callbacks
from pytorch_lightning.callbacks import TQDMProgressBar, RichModelSummary, RichProgressBar, ModelCheckpoint, EarlyStopping
from pl_bolts.callbacks import PrintTableMetricsCallback, TrainingDataMonitor
from utils.utils import get_ckpt_path

from dataloader.data_loader import WalkDataModule
from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from argparse import ArgumentParser

import pytorch_lightning
# %%


def get_parameters():
    '''
    The parameters for the model training, can be called out via the --h menu
    '''
    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'csn', 'r2plus1d', 'x3d', 'slowfast', 'c2d', 'i3d'])
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--version', type=str, default='test', help='the version of logger, such data')
    parser.add_argument('--model_class_num', type=int, default=1, help='the class num of model')
    parser.add_argument('--model_depth', type=int, default=50, choices=[50, 101, 152], help='the depth of used model')

    # Training setting
    parser.add_argument('--max_epochs', type=int, default=50, help='numer of epochs of training')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for the dataloader')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader for load video')
    parser.add_argument('--clip_duration', type=int, default=1, help='clip duration for the video')
    parser.add_argument('--uniform_temporal_subsample_num', type=int,
                        default=8, help='num frame from the clip duration')
    parser.add_argument('--gpu_num', type=int, default=0, choices=[0, 1], help='the gpu number whicht to train')

    # ablation experment 
    # different fusion method 
    parser.add_argument('--fusion_method', type=str, default='slow_fusion', choices=['single_frame', 'early_fusion', 'late_fusion', 'slow_fusion'], help="select the different fusion method from ['single_frame', 'early_fusion', 'late_fusion']")
    
    # Transfor_learning
    parser.add_argument('--transfor_learning', action='store_true', help='if use the transformer learning')
    parser.add_argument('--fix_layer', type=str, default='all', choices=['all', 'head', 'stem_head', 'stage_head'], help="select the ablation study within the choices ['all', 'head', 'stem_head', 'stage_head'].")

    # TTUR
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for optimizer')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    #TODO:Path,　before detection
    parser.add_argument('--data_path_a', type=str, default="/workspace/data/Cross_Validation/ex_20250116_ap_organized", help='meta dataset path')
    parser.add_argument('--data_path_b', type=str, default="/workspace/data/Cross_Validation/ex_20250116_lat_organized", help='meta dataset path')
    
    parser.add_argument('--split_data_path', type=str)
    
    #TODO: change this path, after detection
    parser.add_argument('--split_pad_data_path', type=str, default="/workspace/data/Cross_Validation/ex_20250122_lat",
                        help="split and pad dataset with detection method.")
    parser.add_argument('--seg_data_path', type=str, default="/workspace/data/Cross_Validation/ex_20250122_lat",
                        help="segmentation dataset with mediapipe, with 5 fold cross validation.")

    parser.add_argument('--log_path', type=str, default='./logs', help='the lightning logs saved path')

    # using pretrained
    parser.add_argument('--pretrained_model', type=bool, default=False,
                        help='if use the pretrained model for training.')

    # add the parser to ther Trainer
    # parser = Trainer.add_argparse_args(parser)

    return parser.parse_known_args()

# %%

def train(hparams):

    # set seed
    seed_everything(42, workers=True)

    classification_module = WalkVideoClassificationLightningModule(hparams)

    # instance the data module
    data_module = WalkDataModule(hparams)

    # for the tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(hparams.log_path, hparams.model), name=hparams.log_version, version=hparams.fold)

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=100)
    rich_model_summary = RichModelSummary(max_depth=2)
    rich_progress_bar = RichProgressBar(refresh_rate=hparams.batch_size)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename='{epoch}-{val_loss:.2f}-{val_acc:.4f}',
        auto_insert_metric_name=True,
        monitor="val_acc",
        mode="max",
        save_last=True,
        save_top_k=1,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor='val_acc',
        patience=3,
        mode='max',
    )

    # bolts callbacks
    # table_metrics_callback = PrintTableMetricsCallback()
    monitor = TrainingDataMonitor(log_every_n_steps=25)

    trainer = Trainer(
                      devices=[hparams.gpu_num,],
                      accelerator="gpu",
                      max_epochs=hparams.max_epochs,
                      logger=tb_logger,
                      #   log_every_n_steps=100,
                      check_val_every_n_epoch=1,
                      callbacks=[progress_bar, rich_model_summary, monitor, model_check_point, early_stopping],
                      #   deterministic=True
                      )

    # from the params
    # trainer = Trainer.from_argparse_args(hparams)

    if hparams.pretrained_model:
        trainer.fit(classification_module, data_module, ckpt_path=get_ckpt_path)
    else:
        # training and val
        trainer.fit(classification_module, data_module)

    # trainer.logged_metrics
    # trainer.callback_metrics

    Acc_list = trainer.validate(classification_module, data_module, ckpt_path='best')

    # return the best acc score.
    return model_check_point.best_model_score.item()
 
# %%
if __name__ == '__main__':

    # for test in jupyter
    config, unkonwn = get_parameters()

    #############
    # K Fold CV
    #############

    DATA_PATH_A = config.data_path_a
    DATA_PATH_B = config.data_path_b
    
    # DATA_PATH = config.seg_data_path

    # get the fold number
    print("DATA_PATH_A:", DATA_PATH_A)
    fold_num_a = os.listdir(DATA_PATH_A)
    print("DATA_PATH_B:", DATA_PATH_B)
    fold_num_b = os.listdir(DATA_PATH_B)
    fold_num_a.sort()
    fold_num_b.sort()
    print("fold_num_a:", fold_num_a)
    print("fold_num_b:", fold_num_b)

    
    store_Acc_Dict = {}
    sum_list = []

    for fold in fold_num_a:
        #################
        # start k Fold CV
        #################

        print('#' * 50)
        print('Strat %s' % fold)
        print('#' * 50)

        config.train_path_a = os.path.join(DATA_PATH_A, fold)
        config.train_path_b = os.path.join(DATA_PATH_B, fold)
        config.fold = fold

        # connect the version + model + depth, for tensorboard logger.
        config.log_version = config.version + '_' + config.model + '_depth' + str(config.model_depth)

        Acc_score = train(config)

        store_Acc_Dict[fold] = Acc_score
        sum_list.append(Acc_score)

    print('#' * 50)
    print('different fold Acc:')
    print(store_Acc_Dict)
    print('Final avg Acc is: %s' % (sum(sum_list) / len(sum_list)))
    