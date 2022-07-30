<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="resources/header.png" alt="Logo">
  </a>

<h3 align="center">Northeastern SMILE Lab - Recognizing Faces in the Wild</h3>
  <!--
  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>-->
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About This Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

<!--
[![Product Name Screen Shot][product-screenshot]](https://example.com)
-->

Do you have your father’s nose?

Blood relatives often share facial features. Now researchers at Northeastern University want to improve their algorithm for facial image classification to bridge the gap between research and other familial markers like DNA results. That will be your challenge in this new Kaggle competition.

An automatic kinship classifier has been in the works at Northeastern since 2010. Yet this technology remains largely unseen in practice for a couple of reasons:

1. Existing image databases for kinship recognition tasks aren't large enough to capture and reflect the true data distributions of the families of the world.

2. Many hidden factors affect familial facial relationships, so a more discriminant model is needed than the computer vision algorithms used most often for higher-level categorizations (e.g. facial recognition or object classification).

In this competition, you’ll help researchers build a more complex model by determining if two people are blood-related based solely on images of their faces. If you think you can get it "on the nose," this competition is for you.

Note: The SMILE Lab at Northeastern focuses on the frontier research of applied machine learning, social media analytics, human-computer interaction, and high-level image and video understanding. Their research is driven by the explosion of diverse multimedia from the Internet, including both personal and publicly-available photos and videos. They start by treating fundamental theory from learning algorithms as the soul of machine intelligence and arm it with visual perception.

Source:

* https://www.kaggle.com/c/recognizing-faces-in-the-wild

### Prerequisites

Hardware platform:

* NVIDIA Tesla M40 24GB for inference usage
* AMD 6900XT 16GB for Pytorch ROCm[HIP] for training usage

Software platform:

* NVIDIA SDK: TensorRT/CUDA/CuDNN/APEX
* Python environment: Anaconda
* Deep learning framework: Pytorch
* OS: Arch Linux

<p align="right">(<a href="#top">back to top</a>)</p>

## Workflow

### Model training

The model structure
This model is trained using Pytorch ROCm with NVIDIA/apex.
After training, the weight is saved to the .pt file.

Running time(average on 25 training epochs):

* Training without AMP = 1.364973166 seconds
* Training with AMP = 0.8443774253 seconds
* Improvement: 38.1396% decrease in the running time

Accuracy:

* Training without AMP = 58.52%
* Training with AMP = 59.26%

### Export to ONNX file

Reload the .pt file to construct the original Pytorch model.
Export the pytorch model to the .onnx format using following code snippet.

```python
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# Input to the model
input_data = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
output_data = torch_model(input)
output_file_name = "target.onnx"
# Export the model
torch.onnx.export(torch_model,  # model being run
                  input_data,  # model input (or a tuple for multiple inputs)
                  output_file_name,  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})

```

Once the exporting is done, we can load the .onnx file for validation purpose.
The following snippet is used to load the .onnx file and use it as the inference engine.

```python
import onnx

onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession("super_resolution.onnx")


def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")

```

### Export to TensorRT engine file

Once the conversion and validation for the .onnx file is done,
we can convert it to the TensorRT engine at this point.
Once the TensorRT engine is generated, we run the test again to make sure its precision keeps.

In our experiment setup, all INT8/FP16/TF32/FP32 configuration remains at 99% precision.
Note: If the selected precision is INT8, the calibrator dataset should be provided.

### Engine file size comparison

* classifier-sim.INT8.engine = 6.3M
* classifier-sim.FP16.engine = 3.2M
* classifier-sim.FP32.engine = 3.2M
* classifier-sim.TF32.engine = 3.2M

### Performance evaluation

The performance evaluation is done using a Tesla M40 card.
Corresponding specification can be found here:
https://www.techpowerup.com/gpu-specs/tesla-m40-24-gb.c3838

1. for the INT8 throughput:

```shell
trtexec --loadEngine=siamese_network.INT8.engine --batch=1024 --streams=32 --verbose --avgRuns=16
[07/25/2022-23:41:34] [I] Throughput: 17990.4 qps
[07/25/2022-23:41:34] [I] Latency: min = 125.305 ms, max = 591.657 ms, mean = 436.587 ms, median = 457.4 ms, percentile(99%) = 583.968 ms
[07/25/2022-23:41:34] [I] Enqueue Time: min = 0.0322266 ms, max = 0.111816 ms, mean = 0.0368327 ms, median = 0.0351562 ms, percentile(99%) = 0.108032 ms
[07/25/2022-23:41:34] [I] H2D Latency: min = 5.15454 ms, max = 128.347 ms, mean = 11.8364 ms, median = 5.2063 ms, percentile(99%) = 128.274 ms
[07/25/2022-23:41:34] [I] GPU Compute Time: min = 120.107 ms, max = 518.805 ms, mean = 424.731 ms, median = 451.014 ms, percentile(99%) = 498.792 ms
[07/25/2022-23:41:34] [I] D2H Latency: min = 0.00488281 ms, max = 0.0761719 ms, mean = 0.0197522 ms, median = 0.015625 ms, percentile(99%) = 0.069458 ms
[07/25/2022-23:41:34] [I] Total Host Walltime: 17.9865 s
[07/25/2022-23:41:34] [I] Total GPU Compute Time: 134.215 s
[07/25/2022-23:41:34] [W] * GPU compute time is unstable, with coefficient of variance = 20.2126%.
[07/25/2022-23:41:34] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/25/2022-23:41:34] [I] Explanations of the performance metrics are printed in the verbose logs.
```

2. for the FP16 throughput:

```shell
trtexec --loadEngine=siamese_network.FP16.engine --batch=1024 --streams=32 --verbose --avgRuns=16
[07/25/2022-23:42:50] [I] Throughput: 17836.3 qps
[07/25/2022-23:42:50] [I] Latency: min = 141.965 ms, max = 591.346 ms, mean = 441.273 ms, median = 461.8 ms, percentile(99%) = 586.311 ms
[07/25/2022-23:42:50] [I] Enqueue Time: min = 0.0312753 ms, max = 0.112305 ms, mean = 0.0365971 ms, median = 0.0341797 ms, percentile(99%) = 0.108398 ms
[07/25/2022-23:42:50] [I] H2D Latency: min = 4.97278 ms, max = 128.81 ms, mean = 11.8821 ms, median = 5.21436 ms, percentile(99%) = 128.547 ms
[07/25/2022-23:42:50] [I] GPU Compute Time: min = 136.726 ms, max = 540.65 ms, mean = 429.371 ms, median = 454.791 ms, percentile(99%) = 512.942 ms
[07/25/2022-23:42:50] [I] D2H Latency: min = 0.00585938 ms, max = 0.074585 ms, mean = 0.0199895 ms, median = 0.0159912 ms, percentile(99%) = 0.0653687 ms
[07/25/2022-23:42:50] [I] Total Host Walltime: 18.1418 s
[07/25/2022-23:42:50] [I] Total GPU Compute Time: 135.681 s
[07/25/2022-23:42:50] [W] * GPU compute time is unstable, with coefficient of variance = 20.0353%.
[07/25/2022-23:42:50] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/25/2022-23:42:50] [I] Explanations of the performance metrics are printed in the verbose logs.

```

3. for the TF32 throughput:

```shell
trtexec --loadEngine=siamese_network.TF32.engine --batch=1024 --streams=32 --verbose --avgRuns=16
[07/25/2022-23:44:04] [I] Throughput: 17790.6 qps
[07/25/2022-23:44:04] [I] Latency: min = 157.068 ms, max = 610.103 ms, mean = 441.194 ms, median = 462.976 ms, percentile(99%) = 589.87 ms
[07/25/2022-23:44:04] [I] Enqueue Time: min = 0.03125 ms, max = 0.116211 ms, mean = 0.0366031 ms, median = 0.0341797 ms, percentile(99%) = 0.103027 ms
[07/25/2022-23:44:04] [I] H2D Latency: min = 5.16016 ms, max = 130.113 ms, mean = 11.9496 ms, median = 5.20874 ms, percentile(99%) = 128.493 ms
[07/25/2022-23:44:04] [I] GPU Compute Time: min = 150.438 ms, max = 528.749 ms, mean = 429.226 ms, median = 455.411 ms, percentile(99%) = 507.489 ms
[07/25/2022-23:44:04] [I] D2H Latency: min = 0.00585938 ms, max = 0.0695801 ms, mean = 0.018364 ms, median = 0.015625 ms, percentile(99%) = 0.0638428 ms
[07/25/2022-23:44:04] [I] Total Host Walltime: 18.1885 s
[07/25/2022-23:44:04] [I] Total GPU Compute Time: 135.635 s
[07/25/2022-23:44:04] [W] * GPU compute time is unstable, with coefficient of variance = 20.0225%.
[07/25/2022-23:44:04] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/25/2022-23:44:04] [I] Explanations of the performance metrics are printed in the verbose logs.

```

4. for the FP32 throughput:

```shell
trtexec --loadEngine=siamese_network.FP32.engine --batch=1024 --streams=32 --verbose --avgRuns=16
[07/25/2022-23:44:54] [I] Throughput: 17802 qps
[07/25/2022-23:44:54] [I] Latency: min = 148.669 ms, max = 609.335 ms, mean = 440.522 ms, median = 463.186 ms, percentile(99%) = 591.964 ms
[07/25/2022-23:44:54] [I] Enqueue Time: min = 0.0316539 ms, max = 0.118164 ms, mean = 0.036653 ms, median = 0.0341797 ms, percentile(99%) = 0.100342 ms
[07/25/2022-23:44:54] [I] H2D Latency: min = 5.16895 ms, max = 128.896 ms, mean = 11.7314 ms, median = 5.22095 ms, percentile(99%) = 128.769 ms
[07/25/2022-23:44:54] [I] GPU Compute Time: min = 143.43 ms, max = 521.862 ms, mean = 428.771 ms, median = 456.87 ms, percentile(99%) = 496.774 ms
[07/25/2022-23:44:54] [I] D2H Latency: min = 0.00390625 ms, max = 0.0810547 ms, mean = 0.0191196 ms, median = 0.015625 ms, percentile(99%) = 0.0683594 ms
[07/25/2022-23:44:54] [I] Total Host Walltime: 18.1768 s
[07/25/2022-23:44:54] [I] Total GPU Compute Time: 135.492 s
[07/25/2022-23:44:54] [W] * GPU compute time is unstable, with coefficient of variance = 20.0761%.
[07/25/2022-23:44:54] [W]   If not already in use, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[07/25/2022-23:44:54] [I] Explanations of the performance metrics are printed in the verbose logs.

```

### Accuracy evaluation

* final run loss by classifier-sim.INT8.engine: 18.15721794217825
* final run loss by classifier-sim.FP16.engine: 18.15721845626831
* final run loss by classifier-sim.TF32.engine: 18.157218277454376
* final run loss by classifier-sim.FP32.engine: 18.157218001782894

### Conclusion:

1. The file size of classifier-sim.INT8.engine is almost double as other engine files. This needs further investigation.
2. In term of accuracy, four engines are very close, which means quantization causes negligible accuracy loss.
3. The FP16 quantization has the best performance, which is slightly better than others.

<!-- BUILIT WITH
### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [![Next][Next.js]][TensorRT]
<p align="right">(<a href="#top">back to top</a>)</p>
 -->

<!-- GETTING STARTED
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.
-->


<!-- ROADMAP -->

## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Explorer more possibility

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (
and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge

[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge

[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members

[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge

[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers

[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge

[issues-url]: https://github.com/othneildrew/Best-README-Template/issues

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge

[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/othneildrew

[product-screenshot]: images/screenshot.png

[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white

[Next-url]: https://nextjs.org/

[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB

[React-url]: https://reactjs.org/

[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D

[Vue-url]: https://vuejs.org/

[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white

[Angular-url]: https://angular.io/

[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00

[Svelte-url]: https://svelte.dev/

[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white

[Laravel-url]: https://laravel.com

[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white

[Bootstrap-url]: https://getbootstrap.com

[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white

[JQuery-url]: https://jquery.com

[TensorRT]: https://developer.nvidia.com/tensorrt

<!-- data

-->
