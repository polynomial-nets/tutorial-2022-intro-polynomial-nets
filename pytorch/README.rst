===========================================
Tutorial on high-degree polynomial networks
===========================================

.. image:: https://img.shields.io/badge/PyTorch-1.12.0-red.svg
   :target: https://github.com/polynomial-nets/tutorial-2022-intro-polynomial-nets
   :alt: PyTorch


This code implements two polynomial networks for image recognition **in PyTorch**. 
The two codes are based on the paper of `"**Π-nets: Deep Polynomial Neural Networks**" <https://ieeexplore.ieee.org/document/9353253>`_ (also available `here <https://arxiv.org/abs/2006.13026>`_ ) [1]_.

Those networks aim to demonstrate the performance of the polynomial networks with minimal code examples; therefore, they are not really the state-of-the-art results on recognition. For networks that can achieve state-of-the-art results the source code of the papers can be followed, since they have more intricate implementations. For instance, for Π-nets, please check [1]_.

The two networks include the following: 

*    The jupyter notebook ``CCP_model_minimum_example.ipynb`` implements a simple CCP model on MNIST classification. This can be opened and executed directly in a Google Colab environment.

*    The python files implement a product of polynomials (each polynomial has an NCP-based second degree polynomial). 



Train the network (of the *.py files)
====================================

To train the network, you can execute the following command::

   python train_main.py



Apart from PyTorch, the code depends on Pyaml [2]_.


References
==========

.. [1] https://github.com/grigorisg9gr/polynomial_nets/

.. [2] https://pypi.org/project/pyaml/

