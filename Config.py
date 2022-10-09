from easydict import EasyDict
import time


def get_config():
    conf = EasyDict()
    conf.workspace = './workspace'
    nowdate = time.strftime("%Y%m%d%H", time.localtime())
    image_path = conf.workspace + '/images/' + nowdate
    model_path = conf.workspace + '/models/' + nowdate
    conf.image_path = image_path
    conf.model_path = model_path
    conf.DATAPATH = './workspace/dataset/new'
    conf.MODELPATH = ''
    conf.ISTHETALOSS = True
    conf.sample = 1.0
    conf.train_ratio = 0.7
    conf.batch_size = 4
    conf.epochs = 20
    conf.beam_size = 100
    conf.num_processes = 4
    conf.lr = 0.001
    conf.problimit = 0.0    # 概率下线
    conf.re_str = '^\d{2}(-\d{2,})+_\d{4,}_(211412|211214|211232){1}(-[1-4]{6}){1,}-2331112.png$'
    return conf
