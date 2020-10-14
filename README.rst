Numerical Site Calibration Benchmark: The Alaiz case
====================================================
`Javier Sanz Rodrigo <mailto:jsrodrigo@cener.com>`_, `Gang Huang <mailto:gang.huang@meteodyn.com>`_ 


Background 
----------
This repository holds model evaluation scripts for the `Alaiz Numerical Site Calibration benchmark <https://thewindvaneblog.com/numerical-site-calibration-benchmark-the-alaiz-case-b3767918d812>`_(NSC) to test microscale models in the prediction of flow correction factors following the IEC 61400-12-4.

The project is integrated in the Phase 3 of the `IEA Task 31 Wakebench <https://community.ieawind.org/task31/home>`_ international framework for wind farm modeling and evaluation.

Scope and Objectives
--------------------
The IEC TC 88 committee, “Wind energy generation systems,” initiated the work on a technical report to evaluate the potential application of NSC. The project team (PT12–4) has outlined the current state of the art in numerical flow modeling and summarized existing guidelines and past benchmarking experience on numerical model verification and validation (V&V). Based on the work undertaken, the team identified the important technical aspects for using flow simulation over terrain for wind application as well as the existing open issues including recommendations for further validation through benchmarking tests. The team concluded that further work is needed before a standard for NSC can be issued.

Therefore, this benchmark seeks the following objectives:

* Ascertain the maturity level of state-of-the-art models in the prediction of flow correction factors.
* Define best practices for conducting and documenting numerical site calibration that could be included in the IEC 61400–12–4 text.
* Identify limitations of current datasets for V&V that could be addressed by a dedicated experiment.

Benchmark Guides
----------------
The following blog post is used to guide benchmark participants:

* `Benchmark guide <https://thewindvaneblog.com/numerical-site-calibration-benchmark-the-alaiz-case-b3767918d812>`_  

Data
----
Benchmark input data and simulation data is published open-access in the following data repository: [zenodo dataset]

Installation
------------
We use Jupyter notebooks based on Python 3. We recomend the `Anaconda distribution <https://www.anaconda.com/distribution/>`_ to install python. The libraries used by the notebooks can be installed with 

.. code:: bash

	$ pip install -r requirements.txt

Citation
--------
You can cite the github repo in the following way:

[zenodo github release]

License
-------
Copyright 2020 CENER
Licensed under the GNU General Public License v3.0

Acknowledgements
----------------
The authors would like to thank the benchmark participants for their simulations and in-kind support in fine-tuning the benchmark set-up and evaluation methodology. The benchmark is run under the umbrella of IEA-Wind Task 31.
