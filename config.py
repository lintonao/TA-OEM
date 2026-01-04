
import argparse
import os


def parse_opts(data_set):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    ''' path '''
    parser.add_argument('--datasetName', type=str, default=data_set, help='[mosi, mosei, sims, iemocap]')
    parser.add_argument('--bert_path_en', type=str, default=os.path.join(current_dir, 'pretrain_data', 'bert_en'))
    parser.add_argument('--bert_path_cn', type=str, default=os.path.join(current_dir, 'pretrain_data', 'bert_cn'))
    parser.add_argument('--model_save_dir', type=str, default=os.path.join(current_dir, 'saves', 'save_model'), help='path to save model.')
    parser.add_argument('--res_save_dir', type=str, default=os.path.join(current_dir, 'saves', 'save_fig'), help='path to save fig.')

    ''' model '''
    parser.add_argument('--dataset', type=str, default=data_set)
    assert data_set in ['mosi', 'mosei', 'sims', 'iemocap']
    if data_set == 'mosi':
        parser.add_argument('--data_path', type=str, default=os.path.join(current_dir, 'data_set', 'MOSI'))
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--d_k', type=int, default=32)
        parser.add_argument('--d_v', type=int, default=32)
        parser.add_argument('--n_heads', type=int, default=4)
        parser.add_argument('--d_ff', type=int, default=64)
        parser.add_argument('--drop', type=float, default=0.2)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--train_samples', type=int, default=1283)
        parser.add_argument('--input_size', type=list, default=[74,47,768])
        parser.add_argument('--num_classes', type=int, default=1)
        parser.add_argument('--batch_first', type=bool, default=False)
        parser.add_argument('--file_name', type=str, default='mosi.pth')
        parser.add_argument('--H', type=float, default=3.0)



    if data_set == 'mosei':
        parser.add_argument('--data_path', type=str, default=os.path.join(current_dir, 'data_set', 'MOSEI'))
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--d_k', type=int, default=32)
        parser.add_argument('--d_v', type=int, default=32)
        parser.add_argument('--n_heads', type=int, default=4)
        parser.add_argument('--d_ff', type=int, default=128)
        parser.add_argument('--drop', type=float, default=0.4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--input_size', type=list, default=[74, 35, 768])
        parser.add_argument('--train_samples', type=int, default=16315)
        parser.add_argument('--num_classes', type=int, default=1)
        parser.add_argument('--batch_first', type=bool, default=False)
        parser.add_argument('--file_name', type=str, default='mosei.pth')
        parser.add_argument('--H', type=float, default=3.0)


    if data_set == 'iemocap':
        parser.add_argument('--data_path', type=str, default=os.path.join(current_dir, 'data_set', 'IEMOCAP'))
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--d_k', type=int, default=32)
        parser.add_argument('--d_v', type=int, default=32)
        parser.add_argument('--n_heads', type=int, default=4)
        parser.add_argument('--d_ff', type=int, default=128)
        parser.add_argument('--drop', type=float, default=0.4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--input_size', type=list, default=[74, 35, 300])
        parser.add_argument('--train_samples', type=int, default=2717)
        parser.add_argument('--num_classes', type=int, default=8)
        parser.add_argument('--aligned', type=bool, default=True)
        parser.add_argument('--batch_first', type=bool, default=True)
        parser.add_argument('--file_name', type=str, default='iemocap.pth')


    if data_set == 'sims':
        parser.add_argument('--data_path', type=str, default=os.path.join(current_dir, 'data_set', 'ch_simsv2'))
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--d_k', type=int, default=32)
        parser.add_argument('--d_v', type=int, default=32)
        parser.add_argument('--n_heads', type=int, default=4)
        parser.add_argument('--d_ff', type=int, default=128)
        parser.add_argument('--drop', type=float, default=0.4)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--input_size', type=list, default=[25, 177, 768])
        parser.add_argument('--train_samples', type=int, default=2272)
        parser.add_argument('--num_classes', type=int, default=1)
        parser.add_argument('--batch_first', type=bool, default=False)
        parser.add_argument('--supvised_nums', type=int, default=2272,
                            help='number of supervised data')
        parser.add_argument('--need_normalized', type=bool, default=True)
        parser.add_argument('--use_bert', type=bool, default=True)
        parser.add_argument('--file_name', type=str, default='sims.pth')
        parser.add_argument('--H', type=float, default=1.0)

    args, _ = parser.parse_known_args()
    parser.add_argument('--guide', type=str, default='V', help='T/A/V')
    parser.add_argument('--is_tune', type=bool, default=False, help='tune parameters ?')
    parser.add_argument('--modelName', type=str, default='ta_oem', help='model name')

    ''' Train '''
    parser.add_argument('--device', type=str, default='cuda',
                        help='Specify the device to run. Defaults to cuda, fallsback to cpu')
    parser.add_argument('--train_mode', type=str, default="regression", help='regression / classification')
    parser.add_argument('--learning_rate_bert', type=float, default=1e-5)
    parser.add_argument('--learning_rate_other', type=float, default=1e-4)
    parser.add_argument('--weight_decay_bert', type=float, default=1e-8)
    parser.add_argument('--weight_decay_other', type=float, default=1e-8)
    parser.add_argument('--sch_list', type=list, default=[50, 100, 150])
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--uni_loss_weight', type=float, default=0.5)
    parser.add_argument('--early_stop', type=int, default=20, help='T/A/V')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--save_best_model', type=bool, default=True)
    parser.add_argument('--save_logs', type=bool, default=True)

    ''' USGM '''
    parser.add_argument('--use_usgm', type=bool, default=False, help='is use USGM ?')
    parser.add_argument('--post_fusion_dim', type=int, default=2 * args.hidden_dim)
    parser.add_argument('--post_text_dim', type=int, default=2 * args.hidden_dim)
    parser.add_argument('--post_audio_dim', type=int, default=2 * args.hidden_dim)
    parser.add_argument('--post_video_dim', type=int, default=2 * args.hidden_dim)
    parser.add_argument('--two_classifier', type=bool, default=False)
    parser.add_argument('--excludeZero', type=bool, default=True)
    args = parser.parse_args()

    return args

