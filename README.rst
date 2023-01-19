===========================================
Tutorial on high-degree polynomial networks
===========================================

.. image:: https://img.shields.io/badge/PyTorch-1.12.0-red.svg
   :target: https://github.com/polynomial-nets/tutorial-2022-intro-polynomial-nets
   :alt: PyTorch

.. image:: https://img.shields.io/badge/TensorFlow-2.4.0-green.svg
   :target: https://github.com/polynomial-nets/tutorial-2022-intro-polynomial-nets
   :alt: TensorFlow

.. image:: https://img.shields.io/badge/JAX-0.3.2-blue.svg
   :target: https://github.com/polynomial-nets/tutorial-2022-intro-polynomial-nets
   :alt: JAX


This code implements two polynomial networks for image recognition. 
The two codes are based on the paper of `"**Π-nets: Deep Polynomial Neural Networks**" <https://ieeexplore.ieee.org/document/9353253>`_ (also available `here <https://arxiv.org/abs/2006.13026>`_ ) [1]_.

The two networks are implemented in both PyTorch and TensorFlow (in the folder ``tensorflow``). Those networks aim to demonstrate the performance of the polynomial networks with minimal code examples; therefore, they are not really the state-of-the-art results on recognition. For networks that can achieve state-of-the-art results the source code of the papers can be followed, since they have more intricate implementations. For instance, for Π-nets, please check [1]_.

Please visit the folders of ```pytorch``` or ```tensorflow``` for implementations in PyTorch and TensorFlow respectively. 


.. image:: https://img.shields.io/badge/-New-brightgreen
   :target: https://github.com/polynomial-nets/tutorial-2022-intro-polynomial-nets
   :alt: New

New JAX and Keras implementations for polynomial networks have been added (e.g., ``Minimum_example_JAX.ipynb``).  


Notebooks with polynomial nets on different frameworks
======================================================

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1UJ3l_t387GTWk8nSlr_fX2SNwXuglnNA
   :alt: PyTorch

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1i858yL63kRE5qWn_nMe8cktTFecxAMBQ
   :alt: TensorFlow

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1R3NVusAxDY6hKue-HMqeZBVY6ABLSn08
   :alt: JAX

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1ZyN-tEa6aBYP1QxLU-KVmYnK-5RCY465
   :alt: Keras

The notebooks are the same as the one in the repo and contain minimum examples in PyTorch, TensorFlow, JAX and Keras respectively. 




Acknowledgements
================

We are thankful to Yongtao for the help of converting the code to TensorFlow. 


References
==========

.. [1] https://github.com/grigorisg9gr/polynomial_nets/

.. [2] https://pypi.org/project/pyaml/

