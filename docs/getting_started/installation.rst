.. _installation:

Installation
============

To install stella with pip::

	pip install stella
	

Alternatively you can install the current development version of stella::

        git clone https://github.com/afeinstein20/stella
	cd stella
	python setup.py install

Backend
-------
stella uses Keras Core with the JAX backend.

- Install dependencies (CPU):

	.. code-block:: bash

			pip install -U keras jax jaxlib

- Select the backend before importing ``keras``:

	.. code-block:: bash

			export KERAS_BACKEND=jax
