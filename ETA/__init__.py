from ETA.utils import get_config, mse, mae, mape, rmse, CheckpointManager

config = get_config("./config.yaml")
metrics = [mse, mae, mape, rmse]
loss_function = mse

from ETA.dataset import get_data as Dataset
from ETA.grud import GRUCell as GRUDCell
from ETA.model import Model
