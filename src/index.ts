import { ImageDatasetMeta, Runtime, ScriptContext } from '@pipcook/core';
// @ts-ignore
import download from 'pipcook-downloader';
import * as path from 'path';
import '@tensorflow/tfjs-backend-cpu';

const MODEL_WEIGHTS_NAME = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5';
const MODEL_URL =
  `http://ai-sample.oss-cn-hangzhou.aliyuncs.com/pipcook/models/resnet50_python/${MODEL_WEIGHTS_NAME}`;

const defineModel = async (boa: any, options: Record<string, any>, path: string, inputShape: number[], outputShape: number) => {
  let {
    loss = 'categorical_crossentropy',
    metrics = [ 'accuracy' ],
    learningRate = 0.001,
    decay = 0.05,
    freeze = false
  } = options;

  const { Adam } = boa.import('tensorflow.keras.optimizers');
  const { ResNet50 } = boa.import('tensorflow.keras.applications.resnet50');
  const { GlobalAveragePooling2D, Dropout, Dense } = boa.import('tensorflow.keras.layers');
  const { Model } = boa.import('tensorflow.keras.models');

  await download(MODEL_URL, path);

  let model = ResNet50(
    boa.kwargs({
      include_top: false,
      weights: 'imagenet',
      input_shape: inputShape
    })
  );

  let output = model.output;
  output = GlobalAveragePooling2D()(output);
  output = Dense(1024, boa.kwargs({
    activation: 'relu'
  }))(output);
  output = Dropout(.5)(output);

  const outputs = Dense(outputShape, boa.kwargs({
    activation: 'softmax'
  }))(output);

  model = Model(boa.kwargs({
    inputs: model.input,
    outputs: outputs
  }));

  if (freeze) {
    for (let layer of model.layers.slice(0, -10)) {
      layer.trainable = false;
    }
  }

  model.compile(boa.kwargs({
    optimizer: Adam(boa.kwargs({
      lr: learningRate,
      decay
    })),
    loss: loss,
    metrics: metrics
  }));

  return model;
}

const trainModel = async (tf: any, options: Record<string, any>, model: any, outputShape: number, api: Runtime<any, ImageDatasetMeta>) => {
  const {
    epochs = 1,
    batchSize = 16
  } = options;
  // @ts-ignore
  const meta = await api.dataSource.getDataSourceMeta();
  const trainLength = meta.size.train;
  const batchesPerEpoch = Math.floor(trainLength / batchSize);
  
  for (let i = 0; i < epochs; i++) {
    console.log(`Epoch ${i}/${epochs} start`);
    for (let j = 0; j < batchesPerEpoch; j++) {
      const dataBatch = await api.dataSource.train.nextBatch(batchSize);
      let xs = dataBatch.map((it => it.data));
      let ys = dataBatch.map((it => it.label));
      const xShapes = xs.map(it => it.shape);
      xs = await Promise.all(xs.map((it) => it.data()));
      xs = xs.map((it, idx) => tf.reshape(tf.convert_to_tensor(Array.from(it)), xShapes[idx]));
      ys = tf.one_hot(ys, outputShape);
      xs = tf.stack(xs);
      ys = tf.stack(ys);
      const trainRes = model.train_on_batch(xs, ys);
      if (j % (batchesPerEpoch / 10) == 0) {
        console.log(`Iteration ${i}/${j} result --- loss: ${trainRes[0]} accuracy: ${trainRes[1]}`);
      }
    }
  }

}

const main = async(api: Runtime<any, ImageDatasetMeta>, options: Record<string, any>, context: ScriptContext) => {
  const { boa, workspace } = context;
  const MODEL_PATH = path.join(workspace.cacheDir, MODEL_WEIGHTS_NAME);
  const tf = boa.import('tensorflow');
  // @ts-ignore
  const meta = await api.dataSource.getDataSourceMeta();

  const outputShape = meta.labelMap.length;
  const inputShape = [ parseInt(meta.dimension.x), parseInt(meta.dimension.y), meta.dimension.z ];

  const model = await defineModel(boa, options, MODEL_PATH, inputShape, outputShape);
  await trainModel(tf, options, model, outputShape, api);
  console.log('start saving')
  model.save(path.join(context.workspace.modelDir, 'model.h5'));
}

export default main;
