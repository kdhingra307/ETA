from ETA.utils import get_config, CheckpointManager

config = get_config("./config.yaml")

from ETA.metrics import mse, mae, mape, rmse, direction
from ETA.dataset import get_data as Dataset
from ETA.dcrnn_cell import DCGRUBlock, DCGRUCell
from ETA.model import Model

metrics = [mse, mae, mape, rmse, direction]
loss_function = mse
