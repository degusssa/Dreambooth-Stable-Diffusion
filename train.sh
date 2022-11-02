#!/bin/bash
# Training

# This isn't used for training, just to help you remember what your trained into the model.
project_name="project_name"

# MAX STEPS
# How many steps do you want to train for?
max_training_steps=2020

# Match class_word to the category of the regularization images you chose above.
class_word="person" # typical uses are "man", "person", "woman"

# This is the unique token you are incorporating into the stable diffusion model.
token="blaz"

reg_data_root="regularization_images/person_ddim"

rm -rf training_images/.ipynb_checkpoints
python "main.py" \
 --base configs/stable-diffusion/v1-finetune_unfrozen.yaml \
 -t \
 --actual_resume "model.ckpt" \
 --reg_data_root $reg_data_root \
 -n $project_name \
 --gpus 0, \
 --data_root "training_images" \
 --max_training_steps $max_training_steps \
 --class_word $class_word \
 --token $token \
 --no-test