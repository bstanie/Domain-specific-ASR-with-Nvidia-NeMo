import nemo, nemo_asr
import numpy as np
import scipy.io.wavfile as wave
import torch

from nemo_asr.helpers import post_process_predictions
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import NeuralType, BatchTag, TimeTag, AxisType
from ruamel.yaml import YAML

def offline_inference(config, encoder, decoder, audio_file):
  MODEL_YAML = config
  CHECKPOINT_ENCODER = encoder
  CHECKPOINT_DECODER = decoder
  sample_rate, signal = wave.read(audio_file)

  # get labels (vocab)
  yaml = YAML(typ="safe")
  with open(MODEL_YAML) as f:
    jasper_model_definition = yaml.load(f)
  labels = jasper_model_definition['labels']

  # build neural factory and neural modules
  neural_factory = nemo.core.NeuralModuleFactory(
    placement=nemo.core.DeviceType.GPU,
    backend=nemo.core.Backend.PyTorch)
  data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
    factory=neural_factory,
    **jasper_model_definition["AudioToMelSpectrogramPreprocessor"])

  jasper_encoder = nemo_asr.JasperEncoder(
    feat_in=jasper_model_definition["AudioToMelSpectrogramPreprocessor"]["features"],
    **jasper_model_definition["JasperEncoder"])

  jasper_decoder = nemo_asr.JasperDecoderForCTC(
    feat_in=jasper_model_definition["JasperEncoder"]["jasper"][-1]["filters"],
    num_classes=len(labels))

  greedy_decoder = nemo_asr.GreedyCTCDecoder()

  # load model
  jasper_encoder.restore_from(CHECKPOINT_ENCODER)
  jasper_decoder.restore_from(CHECKPOINT_DECODER)

  # AudioDataLayer
  class AudioDataLayer(DataLayerNM):
    @staticmethod
    def create_ports():
      input_ports = {}
      output_ports = {
        "audio_signal": NeuralType({0: AxisType(BatchTag),
                                    1: AxisType(TimeTag)}),

        "a_sig_length": NeuralType({0: AxisType(BatchTag)}),
      }
      return input_ports, output_ports

    def __init__(self, **kwargs):
      DataLayerNM.__init__(self, **kwargs)
      self.output_enable = False

    def __iter__(self):
      return self

    def __next__(self):
      if not self.output_enable:
        raise StopIteration
      self.output_enable = False
      return torch.as_tensor(self.signal, dtype=torch.float32), \
            torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signal):
      self.signal = np.reshape(signal.astype(np.float32)/32768., [1, -1])
      self.signal_shape = np.expand_dims(self.signal.size, 0).astype(np.int64)
      self.output_enable = True

    def __len__(self):
      return 1

    @property
    def dataset(self):
      return None

    @property
    def data_iterator(self):
      return self

  # Instantiate necessary neural modules
  data_layer = AudioDataLayer()

  # Define inference DAG
  audio_signal, audio_signal_len = data_layer()
  processed_signal, processed_signal_len = data_preprocessor(
    input_signal=audio_signal,
    length=audio_signal_len)
  encoded, encoded_len = jasper_encoder(audio_signal=processed_signal,
                                        length=processed_signal_len)
  log_probs = jasper_decoder(encoder_output=encoded)
  predictions = greedy_decoder(log_probs=log_probs)

  # audio inference
  data_layer.set_signal(signal)

  tensors = neural_factory.infer([
    audio_signal,
    processed_signal,
    encoded,
    log_probs,
    predictions], verbose=False)

  # results
  audio = tensors[0][0][0].cpu().numpy()
  features = tensors[1][0][0].cpu().numpy()
  encoded_features = tensors[2][0][0].cpu().numpy(),
  probs = tensors[3][0][0].cpu().numpy()
  preds = tensors[4][0]
  transcript = post_process_predictions([preds], labels)

  return transcript, audio, features, encoded_features, probs, preds