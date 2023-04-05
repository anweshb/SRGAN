# SRGAN

## This is my implementation of SRGAN to upscale 32x32 images to 128x128 with perceptual loss function

## Sample Results

<!-- ![Input](./Samples/test_input1.png) ![Prediction](./Samples/test_op1.png) ![Ground Truth](./Samples/test_gt1.png) -->

### Example 1

<p align="center">
  <figure>
    <img src="./Samples/test_input1.png" width="30%" />
    <figcaption>Low Resolution Input</figcaption>
  </figure>
  
  <figure>
    <img src="./Samples/test_op1.png" width="30%" /> 
    <figcaption>Prediction</figcaption>
  </figure>
  
  <figure>
    <img src="./Samples/test_gt1.png" width="30%" />
    <figcaption>Groundtruth High Resolution</figcaption>
  </figure>
</p>


### Example 2

<p align="center">
  <figure>
    <img src="./Samples/test_input2.png" width="30%" />
    <figcaption>Low Resolution Input</figcaption>
  </figure>
  
  <figure>
    <img src="./Samples/test_op2.png" width="30%" /> 
    <figcaption>Prediction</figcaption>
  </figure>
  
  <figure>
    <img src="./Samples/test_gt2.png" width="30%" />
    <figcaption>Groundtruth High Resolution</figcaption>
  </figure>
</p>

## It is visible that the prediction isnt as good as the ground truth but better than the input in terms of resolution. However, the colour consistency isn't being maintained in the predictions