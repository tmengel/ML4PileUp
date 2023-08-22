# Performace Plots
The following plots were generated using the `training_history.ipynb` script in the `plotting` directory.
Total model loss, and the loss for each of the three sub-models, are shown below.

## Total Model Loss

![alt text][loss]

[loss]: https://github.com/tmengel/ML4PileUp/blob/main/models/triad_1000e/plots/loss.png 'Total Model Loss'

**Figure 1:** Total model loss for the triad model. The loss is calculated as the sum of the losses for the three sub-models.

## Sub-Model Losses

![alt text][subloss]

[subloss]: https://github.com/tmengel/ML4PileUp/blob/main/models/triad_1000e/plots/submodule_loss.png 'Sub-Model Losses'

**Figure 2:** Losses for the three sub-models. The loss is calculated as the mean squared error between the predicted and true values for each sub-model.



