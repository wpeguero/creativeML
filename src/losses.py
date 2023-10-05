"""Contains custom losses for better generative model training."""
from tensorflow import ones_like, zeros_like
from tensorflow.keras.losses import BinaryCrossentropy, Loss

def _main():
    pass


def discriminator_loss(real_output, fake_output, loss_type:Loss):
    """
    Calculate the loss for the discriminator model.

    -----------------------------------------------
    customized loss function that calculates the loss from the
    fake images and the real images. It does this by first
    comparing the prediction of the discriminator on real images
    to an array of 1s and the discriminator's prediction on fake
    images to an arrays of 0s (The end of the discriminator model
    is a Dense layer with one node that is meant to predict
    whether the image is real (1) or fake (0)).

    Parameter(s)
    ------------
    real_output
        An array built from the real image within the dataset.

    fake_output
        An array built by the generative model.

    loss_type : TensorFlow Loss
        The kind of loss used to calculate the loss of the models.

    Returns
    -------
    total_loss
        The sum of the losses for the real and the fake images.
    """
    real_loss = loss_type(ones_like(real_output), real_output)
    fake_loss = loss_type(ones_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, loss_type:Loss):
    """Calculate the loss of the generative model."""
    return loss_type(ones_like(fake_output), fake_output)


if __name__ == "__main__":
    _main()
