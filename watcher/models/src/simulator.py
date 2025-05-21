from datetime import timedelta, datetime
import pandas as pd
from .model_loaders import build_watcher
from .generation import monte_carlo, make_sim_dataframe
from ...preprocess import preprocess_for_inference


class Simulator:
    """
    High-level interface for running patient trajectory simulations.

    This class wraps a Watcher model and provides a convenient interface for
    generating multiple simulated patient timelines given initial patient information.

    Example:
        .. code-block:: python

            from watcher.models import Simulator

            simulator = Simulator(
                blueprint="path/to/blueprint",
                device = "cuda:0", # Your GPU ID)

    Attributes:
        blueprint (str): Path to the model blueprint directory.
        device (str): GPU device ID.
        model (Watcher): Instantiated Watcher model ready for inference.
    """

    def __init__(self, blueprint: str, device: str):
        """
        Initialize the Simulator with a trained Watcher model.

        Args:
            blueprint (str): Path to the model blueprint directory.
            device (str): Device for running simulations ('cpu' or 'cuda').
        """

        self.model = build_watcher(blueprint=blueprint, train=False)
        self.model = self.model.to(device)
        self.device = device
        self.blueprint = blueprint

    def simulate(
        self,
        db_schema: str,
        patient_id: str,
        n_iter: int,
        timeline_start: str,
        simulation_start: str,
        time_horizon: int,
        max_length: int,
        age_as_timestamp: bool = False,
        stride: int = 64,
        temperature: float = 1.0,
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulations for a single patient.

        This function performs preprocessing, generates multiple simulated timelines,
        and returns the results as a pandas DataFrame.

        Example:
            .. code-block:: python

                from watcher.models import Simulator

                simulator = Simulator(
                        blueprint="path/to/blueprint",   # Path to the model blueprint
                        device="cuda:0",                 # Your GPU ID
                    )
                simulations = simulator.simulate(
                    db_schema="public",
                    patient_id="0001UT",
                    n_iter= 256,                         # Run 256 simulations
                    timeline_start="2011/01/01 00:00",   # Patient history start time
                    simulation_start="2025/04/26 14:57", # Simulation start time
                    time_horizon = 7,                    # Simulation time horizon in days (7 days)
                    max_length= 2000,                    # Maximum sequence length for each simulation
                    temperature = 1.0,


        The details of the simulation results are available at :ref:`simulation_result_table`.

        Args:
            db_schema (str): PostgreSQL schema name containing patient records.
                (If you have not specified a schema, the schema name is expected to be 'public'.)
            patient_id (str): Patient ID in the database.
            n_iter (int): Number of simulation trajectories to generate.
                If you only want one simulation, set this to 1.
                If you are performing the Monte Carlo simulation, set this to a large number (e.g., 256).
            timeline_start (str): Start date for retrieving patient history (format: '%Y/%m/%d %H:%M').
            simulation_start (str): Simulation start time (format: '%Y/%m/%d %H:%M').
            time_horizon (int): Length of simulation in days. Simulation is completed once the time horizon is reached.
            max_length (int): Maximum allowed sequence length for each simulation.
                If the simulation exceeds this length, the simulation result is discarded.
            age_as_timestamp (bool, optional): If True, timestamps are placed in the `age` column instead of patient age.
            stride (int, optional): Stride size for sliding window decoding. Defaults to 64.
            temperature (float, optional): Sampling temperature for stochasticity. Defaults to 1.0.

        Returns:
            pd.DataFrame: A DataFrame containing simulated timelines.
                - Each simulation is labeled by 'patient_id' (e.g., 'simulation0', 'simulation1', etc.).
        """
        # Preprocess args
        time_horizon_td = timedelta(days=time_horizon)
        # Preprocess clinical records
        timeline, catalog_ids, dob = preprocess_for_inference(
            patient_id=patient_id,
            model=self.model,
            start=timeline_start,
            end=simulation_start,
            db_schema=db_schema,
        )
        # Preprocess_time
        horizon_start_td = datetime.strptime(simulation_start, "%Y/%m/%d %H:%M") - dob
        # Monte Carlo
        prod_timelines, prod_catalog_ids = monte_carlo(
            model=self.model,
            timeline=timeline,
            catalog_ids=catalog_ids,
            n_iter=n_iter,
            time_horizon=time_horizon_td,
            horizon_start=horizon_start_td,
            stride=stride,
            temperature=temperature,
            max_length=max_length,
            stop_vocab=None,
            return_generated_parts_only=True,
        )

        if prod_timelines and prod_catalog_ids:
            df = make_sim_dataframe(
                interpreter=self.model.interpreter,
                prod_timelines=prod_timelines,
                prod_catalog_ids=prod_catalog_ids,
            )
            if age_as_timestamp:
                df["age"] = pd.to_timedelta(df["age"])
                df["age"] = df["age"] + dob
                df["age"] = df["age"].dt.strftime("%Y/%m/%d %H:%M")
        else:
            df = pd.DataFrame()

        return df
