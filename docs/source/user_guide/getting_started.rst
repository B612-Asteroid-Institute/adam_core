Getting Started
===============

Installation
------------

You can install ``adam_core`` using pip:

.. code-block:: bash

   pip install adam_core

Basic Usage
----------

Here's a simple example of using ``adam_core`` to work with orbital data:

.. code-block:: python

   import adam_core
   from adam_core.time import Time
   from adam_core.coordinates import CartesianCoordinates
   
   # Create a time object
   time = Time.from_iso("2024-01-01T00:00:00")
   
   # Create cartesian coordinates
   coords = CartesianCoordinates(
       x=[1.0],  # AU
       y=[0.0],
       z=[0.0],
       vx=[0.0],  # AU/day
       vy=[1.0],
       vz=[0.0],
       time=time
   )

This is just a basic example. For more detailed usage, check out the specific guides for each module in the user guide. 