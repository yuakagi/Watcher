
Tutorial
========

.. contents::
   :local:
   :depth: 2

1. Prepare devices
------------------

   You need a server machine with at leaset one GPU.
   We recommend using NVIDIA GPU with Ampere or newer architecture (A100, H100, etc.).

2. Clone repository and run Docker containers
---------------------------------------------

   Steps:

      1. Clone repository

         .. code-block:: bash

            cd /path/to/your/working_dir
            git clone https://github.com/yuakagi/Watcher.git

      2. Configure settings
         There is a file named `.env.example` in the root directory.
         Please **rename this file to .env** and configure the parameters in it.

         Open the `.env` file and set the required environment variables.

      3. Run Docker containers

         .. code-block:: bash

            cd Watcher
            docker compose up

   Now, you can run scripts inside the container:

      .. code-block:: bash

         docker exec watcher-pytorch-1 python3 /code/mnt/path/to/script.py

      Replace `watcher-pytorch-1` with the name of your PyTorch Docker container if different.
      (`PYTHONPATH` is set to `/code`.)

   For time-consuming jobs (i.e., model training), consider using -d option:

      .. code-block:: bash

         docker exec -d watcher-pytorch-1 python3 /code/mnt/path/to/script.py

3. Upload clinical records to database
--------------------------------------

   .. warning::

      - Please note that this Python package performs only minimal data cleaning on preprocessing (e.g., dropping records with missing critical fields, removing duplicates).
      - Therefore, **it is important to pre-clean your data before uploading** (e.g., normalizing laboratory test results and units, mapping medical codes, etc.).

   .. note::
      - You can use **any medical coding system** (e.g., ICD-10, LOINC, ATC, or custom codes like sequential numbers). What matters is the consistency of coding.
      - The database is exposed on the port `${POSTGRES_PORT}`, which you configure in the `.env` file.
      - You can connect directly to this database using external SQL clients for inspection or manual queries if needed.

   Prepare your clinical dataset by referring to :ref:`clinical_records`.

   The `docker-compose.yml` automatically launches a PostgreSQL server container.
   Upload your clinical records into this database using :meth:`watcher.db.init_db_with_csv`.

   This database will serve as the source for both model training and evaluation.

4. Create dataset
-----------------

   .. note::

      If you plan to fine-tune the model later using an `update` dataset,
      please set the argument `update_period` appropriately when creating the dataset.

   Train the model dataset using :meth:`watcher.preprocess.create_dataset`.

   Once the dataset is created, it can be used for both model training.

   You can also retrieve patient IDs used in the dataset (for training, validation, or testing)
   using :meth:`watcher.preprocess.get_patient_ids`.
   
5. Pretrain models
------------------

   Train the model using :meth:`watcher.training.train_watcher`.

6. Perform simulations
----------------------

   Perform simulations using :meth:`watcher.models.Simulator.simulate`.
   The details of the simulation results (pandas DataFrame) are available at :ref:`simulation_result_table`.

   You can list up test patient IDs using :meth:`watcher.preprocess.get_patient_ids` for model evaluation.

   If needed, you may also directly connect to the PostgreSQL database and perform SQL queries to extract patient subsets of interest.

7. [OPTIONAL] Fine-tune models
------------------------------

   The model learns from all training data without weighting. Such training may be suboptimal because medical practices shift over time.
   Therefore, this package allows fine-tuning the model using only the latest data.

   Fine-tune the model using :meth:`watcher.training.train_watcher`.

8. [OPTIONAL] Simulator demo with GUI
-------------------------------------

   You can explore the pretrained model’s inference capabilities using our interactive demo GUI.
   To run the demo, open the notebook at `watcher/notebooks/demo_gui.ipynb`.


9. [OPTIONAL] Use Simulation API
---------------------------------

   .. note::

      - This API is designed to be used as the simulator backend in our digital-twin EHR system (https://github.com/yuakagi/TwinEHR)
      - Please use this API in combination with digital-twin EHR together with this AI

   **Launching the API Server**

   First, please open api_launcher.py, and set the argumets (blueprint, gpu_ids, log_dir, etc.) in the script.
   Then, run the API server using the following command:

   .. code-block:: bash

      docker exec watcher-pytorch-1 gunicorn api_launcher:app --bind 0.0.0.0:63425 --workers 1

   currently, we do not support multiple gunicorn workers. **Ensure that the number of workers is set to 1 ( --workers 1)**.