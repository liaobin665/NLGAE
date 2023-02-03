import argparse

def get_NLGAE_config():
    
    config = argparse.ArgumentParser(description="Run nlgae.")

    config.add_argument('--dataset', nargs='?', default='cora',
                        help='Input dataset')

    config.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate. Default is 0.001.')

    config.add_argument('--n-epochs', default=300, type=int,
                        help='Number of epochs')

    # default=[512, 256],
    config.add_argument('--hidden-dims', type=list, nargs='+', default=[512, 256],
                        help='Number of dimensions.')

    config.add_argument('--lambda-', default=1.0, type=float,
                        help='Parameter controlling the contribution of edge reconstruction in the loss function.')

    config.add_argument('--dropout', default=0.0, type=float,
                        help='Dropout.')

    config.add_argument('--gradient_clipping', default=5.0, type=float,
                        help='gradient clipping')

    return config.parse_args()
