import argparse
import logging
import os

import pandas as pd
import torch
from tensorboardX import SummaryWriter

import core.logger as Logger
import core.metrics as Metrics
import data as Data
import model as Model
from decimal import Decimal


def time_test(params, strategy_params, temp_list):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    opt = params['opt']
    logger = params['logger']
    logger_test = params['logger_test']
    model_epoch = params['model_epoch']

    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    current_step = diffusion.begin_step  # 0
    current_epoch = diffusion.begin_epoch  # 0

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    logger.info('Begin Model Evaluation.')
    idx = 0

    all_datas = pd.DataFrame()
    sr_datas = pd.DataFrame()
    differ_datas = pd.DataFrame()

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    for _, test_data in enumerate(test_loader):

        # ## 这里为了方便调试
        # if idx==1:break
        idx += 1
        diffusion.feed_data(test_data)  # 将数据传入模型
        diffusion.test(continous=False)   # 模型测试
        visuals = diffusion.get_current_visuals() # 获取模型输出的结果

        all_data, sr_df, differ_df = Metrics.tensor2allcsv(visuals, params['col_num'])  # 将模型输出的结果转换为csv格式
        all_datas = Metrics.merge_all_csv(all_datas, all_data) ## 将所有的结果合并
        sr_datas = Metrics.merge_all_csv(sr_datas, sr_df)  # 将所有的结果合并
        differ_datas = Metrics.merge_all_csv(differ_datas, differ_df)  # 将所有的结果合并

    all_datas = all_datas.reset_index(drop=True)
    sr_datas = sr_datas.reset_index(drop=True)
    differ_datas = differ_datas.reset_index(drop=True)

    for i in range(params['row_num'], all_datas.shape[0]):
        all_datas.drop(index=[i], inplace=True)
        sr_datas.drop(index=[i], inplace=True)
        differ_datas.drop(index=[i], inplace=True)

    accuracy,precision,recall, f1 = Metrics.relabeling_strategy(all_datas, strategy_params)

    temp_f1 = Decimal(f1).quantize(Decimal("0.0000"))
    temp_recall = Decimal(recall).quantize(Decimal("0.0000"))
    temp_precision = Decimal(precision).quantize(Decimal("0.0000"))

    print('precision-score: ', float(temp_precision)) # 将这些写到日志文件去
    print('recall-score: ', float(temp_recall))
    print('F1-score: ', float(temp_f1))


# evaluate model performance
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/smap_time_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train ', 'val', 'test'],
                        help='Run either train(training) or val(generation)', default='test')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    temp_list = []
    model_epoch = 100

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args, model_epoch)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    logger_name = 'test' + str(model_epoch)
    # logging
    Logger.setup_logger(logger_name, opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    test_set = Data.create_dataset(opt['datasets']['test'], 'test')

    test_loader = Data.create_dataloader(test_set, opt['datasets']['test'], 'test')
    logger.info('Initial Dataset Finished')
    logger_test = logging.getLogger(logger_name)  # test logger

    start_label = opt['model']['beta_schedule']['test']['start_label']
    end_label = opt['model']['beta_schedule']['test']['end_label']
    step_label = opt['model']['beta_schedule']['test']['step_label']
    step_t = opt['model']['beta_schedule']['test']['step_t']
    strategy_params = {
        'start_label': start_label,  # 1
        'end_label': end_label,  # 3001
        'step_label': step_label, # 1
        'step_t': step_t   # 1000
    }

    params = {
        'opt': opt,
        'logger': logger,
        'logger_test': logger_test,
        'model_epoch': model_epoch,  # 100
        'row_num': test_set.row_num,  # 427617
        'col_num': test_set.col_num  # 15
    }

    time_test(params, strategy_params, temp_list)
    logging.shutdown()

'''
python time_test.py -c config/smap_time_test.json
python time_test.py -c config/msl_time_test.json
python time_test.py -c config/smd_time_test.json

'''