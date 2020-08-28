import time
from pathlib import Path
import pytorch_lightning


def train_batch(inputs, params):
    inputs = inputs.to(params.device)
    params.optimizer.zero_grad()
    outputs = params.model(inputs)
    loss = params.criterion(outputs, inputs)
    loss.backward()
    params.optimizer.step()
    return loss.item()


def test_batch(inputs, params):
    inputs = inputs.to(params.device)
    outputs = params.model(inputs)
    loss = params.criterion(outputs, inputs)
    return loss.item()


def run_flow():

    # Prepare the data
    params = Params()
    train_loader = Data().load_data(dataset=params.train_data, config_mode='train')
    test_loader = Data().load_data(dataset=params.test_data, config_mode='test')
    history = {'epochs': [], 'learning_rate': [], 'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

    runtime_dir = Path('./runtime_dir')
    runtime_dir.mkdir(exist_ok=True, parents=True)
    init_epoch, params.model, history = Utils.get_initial_epoch(model=params.model, runtime_dir=runtime_dir,
                                                                history=history)

    profiler = pytorch_lightning.profiler.SimpleProfiler(output_filename=runtime_dir.joinpath('profile.txt'))

    for epoch in range(init_epoch, params.epochs + 1, 1):

        # Train
        params.model.train()
        train_loss = 0
        epoch_start_time = time.perf_counter()
        profiler.start('train epoch')
        for mini_batch, _ in train_loader:
            profiler.start('train mini-batch')
            # mini_batch = mini_batch.view(mini_batch.shape[0], -1).to(params.device)
            mini_batch = mini_batch.to(params.device)
            train_loss += train_batch(mini_batch, params)
            profiler.stop('train mini-batch')

        train_loss = train_loss / len(train_loader)
        lr = params.scheduler.get_last_lr()
        params.scheduler.step()
        epoch_end_time = time.perf_counter()
        profiler.stop('train epoch')

        # Validate
        profiler.start('validate epoch')
        params.model.eval()
        val_loss = 0
        for mini_batch, _ in test_loader:
            # mini_batch = mini_batch.view(mini_batch.shape[0], -1).to(params.device)
            mini_batch = mini_batch.to(params.device)
            val_loss += test_batch(mini_batch, params)

        val_loss = val_loss / len(test_loader)
        profiler.stop('validate epoch')

        # Log epoch statistics
        history = Utils.update_history(history, epoch, train_loss, val_loss, lr, runtime_dir)
        VisualizationUtils.plot_learning_statistics(history, runtime_dir)
        Utils.save_model_on_epoch_end(epoch, train_loss, val_loss, params.model, runtime_dir)

        print("epoch : {}/{}, train_loss = {:.6f}, val_loss = {:.6f}, time = {:.2f} sec".format(
            epoch, params.epochs, train_loss, val_loss, epoch_end_time - epoch_start_time))

    Utils.save_best_model(runtime_dir, history)

    profiler.describe()
    profiler.summary()


if __name__ == '__main__':
    from configure import Params
    from data import Data
    from visualization_utils import VisualizationUtils
    from utils.utils import Utils

    from torchsummary import summary
    summary(Params().model, (3, 640, 480))
    run_flow()
    ae_predictions_train, ae_predictions_test = VisualizationUtils.visualize_ae_input_output_pairs()

    pass
