# ARMS
Adversarial Robustness of Model Sets

This repository includes basic code to perform the model set attack algorithm  published in the Adversarial Robustness of Model Sets paper at IJCNN 2020.

## Requirements
* python 3.6.8
* keras-gpu 2.2.4
* tensorflow-gpu 1.12.0
* matplotlib 2.2.2
* pillow 6.1.0
* numpy 1.17.3

## Perform model set attacks
Please read the paper or check the config files for the definition of model sets and attack patterns.

After installing the requirements, to perform the attack run the following command:

`python test_attack.py --conf_file config/{pattern_name}/attack_{set_name}.ini`

Avaiable `pattern_name` values:
* consistent
* diverse
* random
* reverse

Avaiable `set_name` values:
* mobiles
* denses
* all


Sample run on the Mobile set using the random pattern:

`python test_attack.py --conf_file config/random/attack_mobiles.ini`

### Custom target pattern
You can define the target pattern for the all set manually in the following config file:
`config/attack_all_custom.ini`.
The `pattern = 2;2;2;2;2;2` line defines the targeted pattern, it is the third class for every member.
Note, the classes are indexed from zero.
