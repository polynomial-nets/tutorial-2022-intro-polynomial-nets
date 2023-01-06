===========================================
Tutorial on high-degree polynomial networks
===========================================

.. image:: https://img.shields.io/badge/PyTorch-1.6.0-red.svg
   :alt: PyTorch

.. image:: https://img.shields.io/badge/TensorFlow-2.4.0-green.svg
   :alt: TensorFlow

.. image:: https://img.shields.io/badge/JAX-0.3.2-blue.svg
   :alt: JAX


This code implements two polynomial networks for image recognition. 
The two codes are based on the paper of `"**Π-nets: Deep Polynomial Neural Networks**" <https://ieeexplore.ieee.org/document/9353253>`_ (also available `here <https://arxiv.org/abs/2006.13026>`_ ) [1]_.

The two networks are implemented in both Pytorch and Tensorflow (in the folder ``tensorflow``). Those networks aim to demonstrate the performance of the polynomial networks with minimal code examples; therefore, they are not really the state-of-the-art results on recognition. For networks that can achieve state-of-the-art results the source code of the papers can be followed, since they have more intricate implementations. For instance, for Π-nets, please check [1]_.

The two networks include the following: 

*    The jupyter notebook ``CCP_model_minimum_example.ipynb`` implements a simple CCP model on MNIST classification. This can be opened and executed directly in a Google Colab environment.

*    The python files implement a product of polynomials (each polynomial has an NCP-based second degree polynomial). 

.. image:: https://img.shields.io/badge/-New-brightgreen
A new JAX implementation for polynomial networks has been added (i.e., ``CCP_minimum_example_JAX.ipynb``).  

 
Train the network (of the *.py files)
====================================

To train the network, you can execute the following command::

   python train_main.py



Apart from PyTorch (or Tensorflow respectively), the code depends on Pyaml [2]_.



Acknowledgements
================

We are thankful to Yongtao for the help of converting the code to Tensorflow. 


References
==========

.. [1] https://github.com/grigorisg9gr/polynomial_nets/

.. [2] https://pypi.org/project/pyaml/

