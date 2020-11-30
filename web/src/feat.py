import argparse
import os
import tempfile

import tensorflow as tf
import tensorflowjs as tfjs

from lidbox.features import audio, cmvn, feature_scaling


@tf.function(input_signature=[
    tf.TensorSpec([None, None, None], tf.float32),
    tf.TensorSpec([], tf.int32),
    tf.TensorSpec([], tf.int32)])
def convertBrowserFFT(spec, sample_rate, num_mel_bins):
    S = audio.db_to_power(spec)
    # S = tf.math.abs(tf.signal.stft(signals, 400, 160, 512))
    # S = audio.spectrograms(signals, sample_rate)
    S = audio.linear_to_mel(S, sample_rate, num_mel_bins=num_mel_bins, fmax=tf.cast(sample_rate/2, tf.float32))
    S = tf.math.log(1e-6 + S)
    S = cmvn(S, axis=1)
    return S

@tf.function(input_signature=[
    tf.TensorSpec([None, None], tf.float32),
    tf.TensorSpec([], tf.int32),
    tf.TensorSpec([], tf.int32)])
def signals2logmel(signals, sample_rate, num_mel_bins):
    signals, sample_rate = signals[:,::3], sample_rate // 3
    flen = audio.ms_to_frames(sample_rate, 25)
    fstep = audio.ms_to_frames(sample_rate, 10)
    S = tf.math.square(tf.math.abs(tf.signal.stft(signals, flen, fstep, fft_length=512)))
    # S = audio.spectrograms(signals, sample_rate)
    S = audio.linear_to_mel(S, sample_rate, num_mel_bins=num_mel_bins, fmax=tf.cast(sample_rate, tf.float32))
    S = tf.math.log(1e-6 + S)
    S = cmvn(S, axis=1)
    return S


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    out_dir = parser.parse_args().out_dir

    export_list = [
        ("spec2logmel", convertBrowserFFT),
        ("signals2logmel", signals2logmel),
    ]

    for name, fn in export_list:
        with tempfile.TemporaryDirectory() as tfmodel_path:
            m = tf.Module()
            m.__call__ = fn
            tf.saved_model.save(m, tfmodel_path)
            tfjs.converters.convert_tf_saved_model(
                    tfmodel_path, os.path.join(out_dir, "tfjs", name))
