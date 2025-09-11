Watcher Documentation
======================

| Watcher is a generative AI model that simulates patient timelines. The AI takes a patient timeline as input and generates future clinical timeline.
| It enables computational modeling of patient trajectories, and synthesizing large-scale patient records.
| Potential applications include personalized medicine, in-silico clinical trials, and counterfactual simulations.

.. image:: _static/generation_video.gif
    :alt: How the model generates patient timelines
    :width: 800px
    :class: center

| Start with 👉 :doc:`Tutorial <tutorial>`
| GitHub 👉 https://github.com/yuakagi/Watcher


Model Overview
======================

.. image:: _static/model_overview.png
    :alt: Watcher model architecture
    :width: 800px
    :class: center

| It encodes categorical, numeric, and temporal entries into vectors, which are processed by decoder-only transformer layers.
| The model autoregressively generates future clinical events, one event at a time.
| At each step, the model outputs a probability distribution over its entire vocabulary.

Digital-Twin EHR System
=======================
| This is our proof-of-concept digital-twin product, and our vision for the future of AI in healthcare.
| Watcher serves as the backend simulator in the system (Figure):

.. image:: _static/digital_twin_system.png
    :alt: Watcher model architecture
    :width: 800px
    :class: center

| This system enables the simulation and real-time feedback through an interactive web application.
| It consists of three components:

1. **AI model** `[GitHub] <https://github.com/yuakagi/Watcher>`_: A generative model that simulates patient timelines.  (👈 This package you are currently reading.)
2. **Digital-twin EHR** `[GitHub] <https://github.com/yuakagi/TwinEHR>`_: A web-based, AI-powered EHR that interacts with the model and visualizes simulation results.  
3. **Data pipeline**: A data pipeline that supplies real-world clinical data to the data server.

To try the full digital-twin system, please follow these steps:

   Step 1: Prepare your clinical data
      - Required clinical data are defined in :ref:`clinical_records`
      - You can use your own clinical data or publicly available datasets.

   Step 2: Upload clinical data to database
      - Watcher package provides a docker container for PostgreSQL database.
      - You can upload your clinical data to the database using the package.

   Step 3: Train the AI model
      - Train (pretrain & fine-tune) the AI model using the Watcher package following the :doc:`tutorial <tutorial>`.

   Step 4: Launch the simulation API server 
      - The Watcher package provides a simulation API server that runs the AI model (gunicorn + Flask).
      - This will be the API server that the digital-twin EHR system will communicate with.
      - Launch the server following the :doc:`tutorial <tutorial>`.

   Step 5: Launch the digital-twin EHR system
      - TwinEHR is a web application that provides a user interface for the simulation API.
      - Clone the repository and set proper environment variables to connect to the simulation API server.
      - Run the web application server

.. note::
   - For Japanese users, `our data pipeline <https://github.com/yuakagi/ssmixtools>`_ is available to conveniently collect and clean clinical data, but its use is not mandatory.
   - Users outside Japan can also use the system with your own clinical data or publicly available datasets.

.. toctree::
   :maxdepth: 3
   :caption: Documentation

   Table Definitions <table_definitions/tables>
   Tutorial <tutorial>

.. toctree::
   :maxdepth: 3
   :caption: Watcher API
   :hidden:

   API <generated/watcher>


