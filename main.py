import Pipeline 
import Models 
import torch
import logging
import numpy as np
from pathlib import Path
from torchfitter.io import save_pickle
from torchfitter.utils.convenience import get_logger
import pandas as pd
from torch.nn.functional import normalize


RESULTS_PATH = Path("results")

logger = get_logger(name="Experiments")
level = logger.level
logging.basicConfig(level=level)

logger.info(f"PROCESSING DATASET")
folder = RESULTS_PATH 
folder.mkdir(exist_ok=True)

### parameters : beta, nglobal, h (length of context vector), window-length, batch_size
params =torch.tensor([ 0.1 , 1 , 1, 20, 64]).float()

### Import dataset and reshape / normalize it 
data =  pd.read_csv('data.csv', sep=",") 
data = data.drop(columns=[ 'Date'])



data = data.astype(dtype="float32")
data = torch.from_numpy(data.values)
data.float()
data = normalize(data, p=2.0)

split_idx = int(data.shape[0] * 0.8)
train_df = data[0:split_idx,:]
remaining_df = data[split_idx: , :]
split_idx_2 = int(remaining_df.size(0) * 0.5)
test_df = remaining_df[0:split_idx_2,:]
validation_df = remaining_df[split_idx_2:, :]

train_df = torch.unsqueeze(train_df, 2)
test_df = torch.unsqueeze(test_df, 2)
validation_df = torch.unsqueeze(validation_df, 2)

### Initialize pipeline with dataset

pipe = Pipeline.DTMLPipeline()
pipe.input_data( [train_df, train_df,validation_df, validation_df, test_df, test_df])

#pip_name = "DTMLpipeline"

#logger.info(f"TRAINING: {pip_name}")

### Create DTML model within pipeline 

pipe.create_model(params)

pipe.model = pipe.model.float()



#logger.info(f"NUMBER OF PARAMS: {count_parameters(pipe.model)}")

### Train the model through the pipeline built-in function
pipe.train_model(params)

### Retrieve predictions, test and history from the training process
y_pred = pipe.preds
y_test = pipe.tests
history = pipe.history

### Save results

np.save(file=folder / "y_pred", arr=y_pred)
np.save(file=folder/ "y_test", arr=y_test)

save_pickle(obj=history, path=folder / "history.pkl")
tr_loss = pipe.history['epoch_history']['loss']['train']
val_loss = pipe.history['epoch_history']['loss']['validation']
import matplotlib.pyplot as plt


plt.plot(tr_loss, label = "training loss")
plt.plot(val_loss, label = "validation loss")
plt.legend(loc='best')
plt.show()

plt.plot(pipe.preds[:,2], label = "prediction")
plt.plot(pipe.tests[:,2], label = "true value")
plt.legend(loc='best')
plt.show()
