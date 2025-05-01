"""
Main entry point for launching Watcher simulators via Flask.
"""

from watcher.watcher_api import start_simulators

# Initialize the Flask app by launching the Watcher simulator backend
# Replace the following paths and settings based on your training output and deployment config.

app = start_simulators(
    blueprint="/path/to/your/watcher_blueprint",  # Path to trained model blueprint directory
    log_dir="/path/to/logs",  # Path to directory for storing logs
    gpu_ids=[  # List of GPU UUIDs to be used
        "GPU-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        "GPU-YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY",
        "GPU-ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ",
    ],
    n_preprocess_workers=2,  # Number of worker processes for preprocessing input (default: 2)
    db_schema="public",  # PostgreSQL schema to use for input/output data (default: 'public')
    max_batch_size=256,  # Max number of simulations per prompt (default: 256)
    max_length=1000,  # Maximum length of timeline to be generated (default: 1000)
    return_generated_parts_only=True,  # Return only generated parts, excluding input prompt
    return_unfinished=False,  # Whether to return sequences that reached max_length cutoff
)

# If running directly (not through Gunicorn), enable development server (optional)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
