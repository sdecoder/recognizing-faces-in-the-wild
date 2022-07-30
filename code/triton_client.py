import numpy as np
import tritonhttpclient
from PIL import Image


def _main():
  _data = _prepare_data()
  _output = _request_server(_data)
  _process_output(_output)
  return


def _process_output(_output):
  print(f'[trace] process your output here')
  pass


def _request_server(_data):
  [req_data0, req_data1] = _data

  # configurations
  VERBOSE = False
  input_name = ['input0', 'input1']
  input_dtype = 'FLOAT32'
  output_name = 'output0'
  model_name = 'model-int8'
  url = 'localhost:8000'
  model_version = '1'
  input_shape = (1, 3, 100, 100)

  input0 = tritonhttpclient.InferInput(input_name[0], input_shape, input_dtype)
  input0.set_data_from_numpy(req_data0, binary_data=False)
  input1 = tritonhttpclient.InferInput(input_name[1], input_shape, input_dtype)
  input1.set_data_from_numpy(req_data1, binary_data=False)
  output = tritonhttpclient.InferRequestedOutput(output_name, binary_data=False)

  triton_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)
  # model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
  # model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)
  response = triton_client.infer(model_name, model_version=model_version, inputs=[input0, input1], outputs=[output])
  output = response.as_numpy(output_name)
  # %% md
  return output

def _prepare_data():
  img0_path = './assets/pic1.jpg'
  img1_path = './assets/pic1.jpg'
  image_pil0 = Image.open(img0_path)
  image_pil1 = Image.open(img1_path)

  from torchvision import transforms

  imagenet_mean = [0.485, 0.456, 0.406]
  imagenet_std = [0.485, 0.456, 0.406]

  resize = transforms.Resize((256, 256))
  center_crop = transforms.CenterCrop(224)
  to_tensor = transforms.ToTensor()
  normalize = transforms.Normalize(mean=imagenet_mean,
                                   std=imagenet_std)

  transform = transforms.Compose([resize, center_crop, to_tensor, normalize])
  image_pil0 = transform(image_pil0).unsqueeze(0).numpy()
  image_pil1 = transform(image_pil1).unsqueeze(0).numpy()
  return [image_pil0, image_pil1]


if __name__ == '__main__':
  _main()
  pass
