import logging
from pathlib import Path

import click
import psutil
from petastorm import make_reader
from petastorm.pytorch import DataLoader
from pytorch_lightning import Trainer

from ml.vae import VAE


@click.command()
@click.option('-d', '--data_path', help='dir path containing model input parquet files', required=True)
@click.option('-m', '--model_path', help='output model path', required=True)
@click.option('--gpu', help='whether to use gpu', default=True, type=bool)
def main(data_path: str, model_path: str, gpu: bool):
    if gpu:
        gpu = -1
    else:
        gpu = None

    # initialise logger
    logger = logging.getLogger(__file__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel('INFO')

    logger.info('Initialise data loader...')
    # get number of cores
    num_cores = psutil.cpu_count(logical=True)
    # load data loader
    reader = make_reader(
        Path(data_path).absolute().as_uri(), reader_pool_type='process', workers_count=num_cores,
        pyarrow_serialize=True, shuffle_row_groups=True, shuffle_row_drop_partitions=2, num_epochs=50
    )
    dataloader = DataLoader(reader, batch_size=300, shuffling_queue_capacity=4096)

    logger.info('Initialise model...')
    # init model
    model = VAE()

    logger.info('Start Training...')
    # train
    trainer = Trainer(val_check_interval=100, max_epochs=1, gpus=gpu)
    trainer.fit(model, dataloader)

    logger.info('Persisting...')
    # persist model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(model_path)

    logger.info('Done')


if __name__ == '__main__':
    main()
