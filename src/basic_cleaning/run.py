#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd  # required for loading/cleaning/saving the CSV


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    logger.info(f"Downloading input artifact: {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Reading CSV from: {artifact_local_path}")
    df = pd.read_csv(artifact_local_path)

    logger.info("Initial rows: %d", len(df))

    # Optional: parse last_review to datetime if present (safe, non-breaking)
    if "last_review" in df.columns:
        logger.info("Parsing 'last_review' column to datetime (if present values are parseable)")
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Filter price outliers using provided arguments
    logger.info("Filtering price between %.2f and %.2f", args.min_price, args.max_price)
    if "price" not in df.columns:
        raise ValueError("Expected a 'price' column in the dataset for outlier filtering.")
    df = df[df["price"].between(args.min_price, args.max_price)].copy()
    logger.info("Rows after price filtering: %d", len(df))

    # Drop rows with any null values
    logger.info("Dropping rows with null values")
    df.dropna(inplace=True)
    logger.info("Rows after dropping nulls: %d", len(df))

    # Save cleaned data
    output_csv = "clean_sample.csv"
    logger.info(f"Saving cleaned data to {output_csv} (index=False)")
    df.to_csv(output_csv, index=False)

    # Log cleaned artifact to W&B
    logger.info(
        "Uploading cleaned artifact to W&B: name=%s, type=%s",
        args.output_artifact, args.output_type
    )
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_csv)
    run.log_artifact(artifact)

    logger.info("Artifact logged successfully.")
    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully qualified name of the input W&B artifact to download (e.g., 'sample.csv:latest')",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the output artifact that will store the cleaned dataset",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Artifact type for the output dataset (e.g., 'clean_sample')",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Human-readable description of the output artifact’s contents",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price threshold used to filter outliers from the dataset",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price threshold used to filter outliers from the dataset",
        required=True
    )


    args = parser.parse_args()

    go(args)
