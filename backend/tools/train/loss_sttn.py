import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    Implemented according to paper https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        Loss types available: 'nsgan' | 'lsgan' | 'hinge'
        type: Specify which type of GAN loss to use.
        target_real_label: Target label value for real images.
        target_fake_label: Target label value for fake images.
        """
        super(AdversarialLoss, self).__init__()
        self.type = type  # Loss type
        # Register labels using buffer so they are saved and loaded with the model
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        # Initialize different loss functions based on selected type
        if type == 'nsgan':
            self.criterion = nn.BCELoss()  # Binary cross entropy loss (non-saturating GAN)
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()  # Mean squared error loss (least squares GAN)
        elif type == 'hinge':
            self.criterion = nn.ReLU()  # ReLU function suitable for hinge loss

    def __call__(self, outputs, is_real, is_disc=None):
        """
        Call function to calculate loss.
        outputs: Network output.
        is_real: True if real sample; False if fake sample.
        is_disc: Indicates whether discriminator is currently being optimized.
        """
        if self.type == 'hinge':
            # For hinge loss
            if is_disc:
                # If discriminator
                if is_real:
                    outputs = -outputs  # Reverse label for real samples
                # max(0, 1 - (real/fake) example output)
                return self.criterion(1 + outputs).mean()
            else:
                # If generator, -min(0, -output) = max(0, output)
                return (-outputs).mean()
        else:
            # For nsgan and lsgan loss
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs)
            # Calculate loss between model output and target labels
            loss = self.criterion(outputs, labels)
            return loss
