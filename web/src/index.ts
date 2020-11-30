import './style.css';
import * as tf from '@tensorflow/tfjs';
// require('@tensorflow/tfjs-backend-wasm');


interface AppState {
  TFBackend: string,
  microphoneConfig: tf.data.MicrophoneConfig;
  microphoneCaptureIntervalMs: number;
  signalCanvas: HTMLCanvasElement;
  // dBSpecCanvas: HTMLCanvasElement;
  logMelSpecCanvas: HTMLCanvasElement;
  predictionLabel: HTMLElement;
  running: boolean;
  renderIntervalID: number;
  spec2logmel: tf.GraphModel;
  signals2logmel: tf.GraphModel;
  xvector: tf.LayersModel;
  int2label: tf.Tensor1D;
}


export const state: AppState = {
  TFBackend: "webgl",
  microphoneConfig: {
    fftSize: 1024,
    numFramesPerSpectrogram: 198,
    sampleRateHz: 44100,
    includeSpectrogram: true,
    includeWaveform: true,
  },
  microphoneCaptureIntervalMs: 200,
  signalCanvas: null,
  // dBSpecCanvas: null,
  logMelSpecCanvas: null,
  predictionLabel: null,
  running: false,
  renderIntervalID: 0,
  spec2logmel: null,
  signals2logmel: null,
  xvector: null,
  int2label: null,
}


function fatalError(error: Error): void {
  console.error(error)
  console.error("cannot recover from error")
  stopApp()
}

export function stopApp(): void {
  console.info("app stopping")
  state.running = false
  if (state.renderIntervalID > 0) {
    window.clearInterval(state.renderIntervalID)
    state.renderIntervalID = 0
  }
}


function createElement(id, tag): HTMLElement {
  const e: HTMLElement = document.createElement(tag)
  e.id = id
  document.body.appendChild(e)
  return e
}


function spectrogramToCanvas(spec: tf.Tensor3D, canvas: HTMLCanvasElement): void {
  // Scale all values between 0 and 1
  const min: tf.Tensor3D = spec.min([0], true)
  const max: tf.Tensor3D = spec.max([0], true)
  let image: tf.Tensor3D = tf.divNoNan(spec.sub(min), max.sub(min))
  image = image.transpose([1, 0, 2]).reverse(0) as tf.Tensor3D

  // Render to canvas
  tf.browser.toPixels(image, canvas).catch(fatalError)
}


let spec2logmelInput = {
  spec: tf.zeros([1, 1, 1]),
  sample_rate: tf.scalar(16000, "int32"),
  num_mel_bins: tf.scalar(40, "int32"),
}

let signals2logmelInput = {
  signals: tf.zeros([1, 1]),
  sample_rate: tf.scalar(state.microphoneConfig.sampleRateHz, "int32"),
  num_mel_bins: tf.scalar(40, "int32"),
}

function updatePredictionLabel(predictedIndexes: Int32Array): void {
  const labels: string[] = Array.from(predictedIndexes, i => state.int2label[i])
  state.predictionLabel.innerText = "prediction: " + labels.join(", ")
}

function handleMicrophoneInput(data: any): void {
  if (!state.running) {
    console.warn("app not running, ignoring microphone input data")
    return
  }
  // tf.tidy(() => spectrogramToCanvas(data.spectrogram.clipByValue(-200, 0), state.dBSpecCanvas))

  tf.tidy(() => {
    spec2logmelInput.spec = data.spectrogram.transpose([2, 0, 1])
    data.spectrogram.dispose()
    const logmel = state.spec2logmel.execute(spec2logmelInput)
    const imgInput = (logmel as tf.Tensor).clipByValue(-1, 1).transpose([1, 2, 0]) as tf.Tensor3D
    spectrogramToCanvas(imgInput, state.logMelSpecCanvas)

    const prediction: tf.Tensor1D = state.xvector.predict(logmel) as tf.Tensor1D
    prediction.argMax(1).data().then(updatePredictionLabel)
  })

  // signal2logmelInput.signals = data.waveform.transpose([1, 0])
  // data.waveform.dispose()

  // state.signals2logmel.executeAsync(signal2logmelInput)
  // .then(logmel => {
  // 		tf.tidy(() => {
  // 			const imgInput = (logmel as tf.Tensor).clipByValue(-1, 1).transpose([1, 2, 0]) as tf.Tensor3D
  // 			spectrogramToCanvas(imgInput, state.logMelSpecCanvas)
  // 			signal2logmelInput.signals.dispose();
  // 			(logmel as tf.Tensor).dispose()
  // 		})
  // 	})
  // .catch(fatalError)

}


function startListenLoop(mic: any): void {
  state.renderIntervalID = window.setInterval(
    () => {
      mic.capture()
      .then(micData => handleMicrophoneInput(micData))
      .catch(fatalError)
    },
    state.microphoneCaptureIntervalMs)
}


async function main() {
  // state.signalCanvas = createCanvas("signal-canvas")
  // state.dBSpecCanvas = createCanvas("decibel-spectrogram-canvas")
  state.logMelSpecCanvas = createElement("logscale-melspectrogram-canvas", "canvas") as HTMLCanvasElement
  state.predictionLabel = createElement("prediction-label", "h2")

  state.microphoneConfig.columnTruncateLength = Math.round(
    (state.microphoneConfig.fftSize / 2 + 1)
    / (state.microphoneConfig.sampleRateHz/16000))

  await tf.setBackend(state.TFBackend)
  console.log("initialized tensorflow.js backend:", tf.getBackend())

  console.log("requesting access to an input device")
  const mic = await tf.data.microphone(state.microphoneConfig)
  console.log("got permission to use input device", (mic as any).stream.id)

  const graph1 = await tf.loadGraphModel("./static/tfjs/spec2logmel/model.json")
  console.log("tf graph1 loaded")
  state.spec2logmel = graph1

  // const graph2 = await tf.loadGraphModel("./static/tfjs/signals2logmel/model.json")
  // console.log("tf graph2 loaded")
  // state.signals2logmel = graph2

  const graph3 = await tf.loadLayersModel("./static/tfjs/xvector_mv/model.json")
  console.log("tf graph3 loaded")
  state.xvector = graph3
  state.xvector.summary()

  const int2label = await tf.util.fetch("./static/tfjs/xvector_mv/int2label.json")
  state.int2label = await int2label.json()

  console.log("starting app")
  state.running = true
  startListenLoop(mic)

}


document.addEventListener("DOMContentLoaded", () => main().catch(fatalError))
