import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import PIL
import timm
import torch
import torchvision
from prettytable import PrettyTable

from domainbed import hparams_registry
from domainbed.datasets import get_dataset
from domainbed.eval import eval_en
from domainbed.lib import misc
from domainbed.lib.logger import Logger
from domainbed.lib.writers import get_writer
from domainbed.trainer import train
from sconf import Config

sys.path.append("../../")
from helpers.wandb import WandbLogger  # Import WandbLogger


def main():
    parser = argparse.ArgumentParser(description="Domain generalization", allow_abbrev=False)
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="datadir/")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and random_hparams).",
    )
    parser.add_argument("--r", type=int, default=4, help="Rank of adapter.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--attention", type=bool, default=None)
    parser.add_argument("--l_aux", action="store_true", help="Use auxiliary loss")
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model_save", default=None, type=int, help="Model save start step")
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",
    )
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    parser.add_argument("--en", type=bool, default=None)
    # Add WandB arguments
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="domain_generalization")
    parser.add_argument("--wandb_entity", type=str, default="UBC-LEAVES")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb_offline", action="store_true", help="Use WandB in offline mode")

    args, left_argv = parser.parse_known_args()
    args.deterministic = True

    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    # hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,args.trial_seed)

    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)

    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.name}"

    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)

    args.out_root = args.work_dir / Path("train_output") / args.dataset
    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(exist_ok=True, parents=True)

    writer = get_writer(args.out_root / "runs" / args.unique_name)
    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")
    # Initialize WandB logger if enabled
    wandb_logger = None
    if args.wandb:
        # Prepare configuration for WandB
        config = {
            "algorithm": args.algorithm,
            "dataset": args.dataset,
            "test_envs": args.test_envs,
            "seed": args.seed,
            "trial_seed": args.trial_seed,
            "adapter_rank": args.r,
            "steps": args.steps,
            "attention_only": args.attention,
            "auxiliary_loss": args.l_aux,
            "holdout_fraction": args.holdout_fraction,
        }

        # Add hyperparameters to config
        for k, v in hparams.items():
            config[f"hparam/{k}"] = v

        # Initialize WandB logger
        wandb_logger = WandbLogger(
            config=config,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.unique_name,
            tags=args.wandb_tags,
            job_type="Training",
            offline=args.wandb_offline,
            notes=f"Algorithm: {args.algorithm}, Dataset: {args.dataset}",
        )
        logger.info(f"WandB initialized: {args.wandb_project}/{args.unique_name}")
    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.nofmt("\tNumPy: {}".format(np.__version__))
    logger.nofmt("\tPIL: {}".format(PIL.__version__))
    logger.nofmt("\ttimm: {}".format(timm.__version__))

    # Different to DomainBed, we support CUDA only.
    assert torch.cuda.is_available(), "CUDA is not available"

    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))

    logger.nofmt("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.nofmt("\t" + line)

    if args.show:
        exit()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    # Dummy datasets for logging information.
    # Real dataset will be re-assigned in train function.
    # test_envs only decide transforms; simply set to zero.
    dataset, _in_splits, _out_splits = get_dataset([0], args, hparams)

    # print dataset information
    logger.nofmt("Dataset:")
    logger.nofmt(f"\t[{args.dataset}] #envs={len(dataset)}, #classes={dataset.num_classes}")
    for i, env_property in enumerate(dataset.environments):
        logger.nofmt(f"\tenv{i}: {env_property} (#{len(dataset[i])})")
    logger.nofmt("")

    # Log dataset info to WandB
    if wandb_logger:
        env_data = []
        for i, env_property in enumerate(dataset.environments):
            env_data.append([i, env_property, len(dataset[i])])
        wandb_logger.log_table(
            "dataset_environments", data=env_data, columns=["Environment ID", "Environment", "Sample Count"]
        )

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")

    org_n_steps = n_steps
    n_steps = (n_steps // checkpoint_freq) * checkpoint_freq + 1
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")

    if not args.test_envs:
        args.test_envs = [[te] for te in range(len(dataset))]
    logger.info(f"Target test envs = {args.test_envs}")

    ###########################################################################
    # Run
    ###########################################################################
    all_records = []
    results = collections.defaultdict(list)

    for test_env in args.test_envs:
        if wandb_logger:
            # Log current test environment
            wandb_logger.log({"current_test_env": test_env[0]})
        res, records = train(
            test_env,
            args=args,
            hparams=hparams,
            n_steps=n_steps,
            checkpoint_freq=checkpoint_freq,
            logger=logger,
            writer=writer,
            wandb_logger=wandb_logger,  # Pass wandb_logger to the train function
        )

        all_records.append(records)
        for k, v in res.items():
            results[k].append(v)

        # Log test results to WandB
        if wandb_logger:
            for k, v in res.items():
                wandb_logger.log({f"test_env_{test_env[0]}/{k}": v})

    # log summary table
    logger.info("=== Summary ===")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("Unique name: %s" % args.unique_name)
    logger.info("Out path: %s" % args.out_dir)
    logger.info("Algorithm: %s" % args.algorithm)
    logger.info("Dataset: %s" % args.dataset)

    table = PrettyTable(["Selection"] + dataset.environments + ["Avg."])
    summary_data = []
    for key, row in results.items():
        row_with_avg = row.copy()
        avg_value = np.mean(row)
        row_with_avg.append(avg_value)

        # Format for console display
        row_formatted = [f"{acc:.3%}" for acc in row_with_avg]
        table.add_row([key] + row_formatted)

        # Prepare data for WandB table
        summary_row = [key] + [float(acc) for acc in row_with_avg]
        summary_data.append(summary_row)
    logger.nofmt(table)

    # Log summary table to WandB
    if wandb_logger:
        columns = ["Selection"] + dataset.environments + ["Average"]
        wandb_logger.log_table("results_summary", data=summary_data, columns=columns)

        # Log final averages
        for key, row in results.items():
            wandb_logger.log({f"final/{key}_avg": np.mean(row)})

        # Save the final model as artifact
        if args.out_dir:
            final_model_path = args.out_dir / "final_model.pth"
            if final_model_path.exists():
                wandb_logger.log_artifact(
                    str(final_model_path),
                    artifact_name="final_model",
                    artifact_type="model",
                    description=f"Final {args.algorithm} model trained on {args.dataset}",
                )

        # Finish the wandb run
        wandb_logger.finish_run()


if __name__ == "__main__":
    main()
