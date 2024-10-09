Autoencoder using Convolutional neural network (CNN)

Author: T M Feroz Ali



Network architecture:

    Autoencoder and Decoder uses 3 layer n/w.

    Encoder ch: (3->32->64-> fc layer (64X3X3 -> 9))
                  img_size: 28 -> 11 -> 3 -> fc
                      
    Decoder ch: (3->32->64-> fc layer (64X3X3 -> 9))
                  img_size: fc -> 3 -> 12 -> 28 
    
    Latent space dimn: 9 
    
    All RelU non-linearity except Sigmoid for output layer
    


Other details:

    Dataset: MNIST

    Loss: BCELoss

    Analyzes latent space using PCA






Plot of train and test losses:

![alt text](https://github.com/ferozalitm/AutoEncoder_CNN/blob/main/Results/Loss.png)



Reconstructrion on training-set samples:

![alt text](https://github.com/ferozalitm/AutoEncoder_CNN/blob/main/Results/train_reconst-149.png)



Reconstructrion on test-set samples:

![alt text](https://github.com/ferozalitm/AutoEncoder_CNN/blob/main/Results/test_reconst-149.png)



Analysis of VAE latent space using PCA:

![alt text](https://github.com/ferozalitm/AutoEncoder_CNN/blob/main/Results/PCA_latentSpace-149.png)


New data generated/sampled from the VAE latent space:

Interpolation between two digits:

![alt text](https://github.com/ferozalitm/AutoEncoder_CNN/blob/main/Results/sampled_3_8-149.png)

![alt text](https://github.com/ferozalitm/AutoEncoder_CNN/blob/main/Results/sampled_1_7-149.png)

Interpolation between four digits:

![alt text](https://github.com/ferozalitm/AutoEncoder_CNN/blob/main/Results/sampled_1_7_3_8-149.png)

