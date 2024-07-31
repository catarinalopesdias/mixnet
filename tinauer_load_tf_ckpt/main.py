import tensorflow as tf
import numpy as np
import h5py

from network_adaptedfrom_BOLLMAN_inputoutput import build_CNN_BOLLMANinputoutput
#from newGA import GradientAccumulateModel

save_path = 'Bg_BollmannExtralayer_newadam16cp-0820_trainsamples500_datasetiter5000_batchsize1_gaaccum10_loss_mse_001_val_loss_unif02_datagen_evenlessbgnoartifacts_ExtraLayer_artif_1_nowrappingCircEager.ckpt'


input_shape = (160, 160, 160, 1)
input_tensor = tf.keras.Input(shape = input_shape, name='input')
output_tensor = build_CNN_BOLLMANinputoutput(input_tensor)

model = tf.keras.Model(input_tensor, output_tensor)
# model = GradientAccumulateModel(accum_steps=10,
#                                 inputs=model.input, outputs=model.output)

#model.summary()

test_data = np.ones((1,) + input_shape)
test_1 = model.predict(test_data)
test_2 = model.predict(test_data)
#print(np.all(np.equal(test_1-test_2, test_2)))

epsilon =1e-5

# runs same model twice with the same dataset
print(np.all((np.abs(test_1-test_2) < epsilon)))


# Manages saving/restoring trackable values to disk.
checkpoint = tf.train.Checkpoint(model)

# runs same model twice with the same dataset

# Restore the checkpointed values to the `model` object.
status = checkpoint.restore(save_path)
#status.expect_partial()

# the model has now new checkpoints

#run the model twice with the same data set
test_3 = model.predict(test_data)
test_4 = model.predict(test_data)
print(np.all((np.abs(test_3-test_4) < epsilon)))
#it showld be true

print(np.all((np.abs(test_1-test_3) < epsilon)))

model.save_weights('test.weights.h5')
