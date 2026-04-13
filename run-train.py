import argparse
import logging
import os
import subprocess
import shutil


def is_windows() -> bool:
    return os.name == "nt"


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("name", help="Model name")
    parser.add_argument("-g", "--gpu", help="GPU device", default="0")
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset path",
    )
    parser.add_argument(
        "-s",
        "--sample-rate",
        type=int,
        choices=[24000, 32000, 40000, 48000],
        help="Sample rate",
        required=True,
    )
    parser.add_argument("-e", "--epochs", type=int, help="Epoch count")
    parser.add_argument(
        "-b", "--batch-size", type=int, help="Batch size", required=True
    )
    parser.add_argument(
        "-f",
        "--epoch-save-frequency",
        type=int,
        help="Epoch save frequency",
        default=10,
    )
    parser.add_argument("-G", "--pretrain-g", help="Pretrain G model link or path")
    parser.add_argument("-D", "--pretrain-d", help="Pretrain D model link or path")
    parser.add_argument(
        "-p",
        "--pretrain",
        help="Pretrain to download",
        choices=["og", "lc1.5", "lc1.6", "klm"],
    )
    parser.add_argument(
        "-r", "--restore-dir", help="If provided, this directory will be restored first"
    )
    parser.add_argument(
        "-n",
        "--normalization-mode",
        choices=["none", "post_peak", "post_peak_rvc", "post_rms"],
        help="Normalization mode",
        default="post_rms",
    )
    parser.add_argument(
        "-m", "--mutes", type=int, default=2, help="Amount of silence clips"
    )
    parser.add_argument(
        "-z",
        "--zip",
        action="store_true",
        help="If set, the output directory will be zipped at the end",
    )
    parser.add_argument(
        "--nh",
        "--no-high-pass",
        dest="high_pass",
        action="store_false",
        help="Disable high-pass filter",
    )
    parser.add_argument(
        "--np",
        "--no-preprocess",
        dest="preprocess",
        action="store_false",
        help="Disable preprocess step",
    )
    parser.add_argument(
        "--ne",
        "--no-extract",
        dest="extract",
        action="store_false",
        help="Disable extract step",
    )
    parser.add_argument(
        "--ni",
        "--no-index",
        dest="index",
        action="store_false",
        help="Disable index step",
    )
    parser.add_argument(
        "--nt",
        "--no-train",
        dest="train",
        action="store_false",
        help="Disable train step",
    )
    args = parser.parse_args()
    if args.dataset and not os.path.isdir(args.dataset):
        parser.error(f"Dataset path '{args.dataset}' doesn't exist")
    if args.preprocess or args.extract or args.index:
        if not args.dataset:
            parser.error(
                "-d/--dataset must be provided for preprocess/extract/index step"
            )
    if args.pretrain_g or args.pretrain_d:
        if args.pretrain:
            parser.error(
                "Pretrain G/D should not be provided when --pretrain is specified"
            )
        if not args.pretrain_g or not args.pretrain_d:
            parser.error(
                "If a custom pretrain is used, both G and D should be provided"
            )
    if args.train:
        if not args.epochs:
            parser.error("-e/--epochs must be provided for training")
        if not args.epoch_save_frequency:
            parser.error("-f/--epoch-save-frequency must be provided for training")
    if args.restore_dir and not os.path.isdir(args.restore_dir):
        parser.error(f"Restore directory '{args.restore_dir}' doesn't exist")
    return args


def get_pretrain_links(pretrain: str, sample_rate: int) -> tuple:
    if pretrain == "lc1.5":
        g = f"https://huggingface.co/lyery/mode4/resolve/main/G_{(15 if sample_rate == 32000 else f'{sample_rate // 1000}k')}.pth?download=true"
    elif pretrain == "lc1.6":
        if sample_rate != 32000:
            raise ValueError(
                f"Sample rate {sample_rate} is not supported by pretrain {pretrain}"
            )
        g = "https://huggingface.co/lyery/legacy_core1.6/resolve/main/G_11.pth?download=true"
    elif pretrain == "klm":
        g = f"https://huggingface.co/SeoulStreamingStation/KLM_RVC_KLM-HF_Trainer/resolve/main/G_KLM_RVC_PT_{sample_rate // 1000}k.pth?download=true"
    else:
        raise ValueError(f"Unknown pretrain name: {pretrain}")
    d = g.replace("/G_", "/D_")
    return g, d


def main():
    args = parse_args()
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s][%(funcName)s:%(lineno)d] %(message)s",
        level=logging.DEBUG,
        datefmt="%H:%M:%S",
    )
    model = (
        f"{args.name}_{args.sample_rate // 1000}k_{args.batch_size}b"
        + (
            f"_{args.normalization_mode}"
            if args.normalization_mode != "post_rms"
            else ""
        )
        + ("_nhp" if not args.high_pass else "")
    )
    logs_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "logs", model))
    logging.info(f"Output path: {logs_path}")
    if args.restore_dir:
        if os.path.isdir(logs_path):
            shutil.rmtree(logs_path)
        logging.info(f"Restoring directory '{args.restore_dir}'...")
        shutil.copytree(args.restore_dir, logs_path)
    if args.pretrain and args.pretrain != "og":
        logging.info(f"Preparing download links for {args.pretrain}...")
        args.pretrain_g, args.pretrain_d = get_pretrain_links(
            args.pretrain, args.sample_rate
        )
    if args.train and args.pretrain_g:
        if os.path.isfile(args.pretrain_g):
            g_path = args.pretrain_g
        else:
            g_path = "pretrain_G.pth"
            logging.info("Downloading custom generator...")
            p = subprocess.run(["wget", "-O", g_path, args.pretrain_g])
            p.check_returncode()

        if os.path.isfile(args.pretrain_d):
            d_path = args.pretrain_d
        else:
            d_path = "pretrain_D.pth"
            logging.info("Downloading custom discriminator...")
            p = subprocess.run(["wget", "-O", d_path, args.pretrain_d])
            p.check_returncode()
    if is_windows():
        CMD_BASE = ["env/python.exe", "core.py"]
    else:
        CMD_BASE = ["uv", "run", "core.py"]
    if args.preprocess:
        cmd_preprocess = CMD_BASE + [
            "preprocess",
            "--model_name",
            model,
            "--dataset_path",
            args.dataset,
            "--sample_rate",
            str(args.sample_rate),
            "--normalization_mode",
            args.normalization_mode,
            "--process_effects",
            str(args.high_pass),
        ]
        logging.info("Preprocessing...")
        p = subprocess.run(cmd_preprocess)
        p.check_returncode()
    if args.extract:
        cmd_extract = CMD_BASE + [
            "extract",
            "--model_name",
            model,
            "--sample_rate",
            str(args.sample_rate),
            "--gpu",
            args.gpu,
            "--include_mutes",
            str(args.mutes),
        ]
        logging.info("Extracting...")
        p = subprocess.run(cmd_extract)
        p.check_returncode()
    if args.index:
        cmd_index = CMD_BASE + ["index", "--model_name", model]
        logging.info("Generating index...")
        p = subprocess.run(cmd_index)
        p.check_returncode()
    if args.train:
        cmd_train = CMD_BASE + [
            "train",
            "--model_name",
            model,
            "--epoch_save_frequency",
            str(args.epoch_save_frequency),
            "--total_epoch_count",
            str(args.epochs),
            "--sample_rate",
            str(args.sample_rate),
            "--save_only_latest_net_models=True",
            "--batch_size",
            str(args.batch_size),
        ]
        if args.pretrain_g:
            cmd_train += [
                "--custom_pretrained=True",
                "--g_pretrained_path",
                g_path,
                "--d_pretrained_path",
                d_path,
            ]
        logging.info("Training...")
        p = subprocess.run(cmd_train)
        p.check_returncode()
    if args.zip:
        logging.info("Zipping...")
        shutil.make_archive(logs_path, "zip", logs_path)
    logging.info("Done!")


if __name__ == "__main__":
    main()
