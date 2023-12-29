import os
import sys
import hydra
import time
from logging import getLogger

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from roberta_finetuning import train
from roberta_pred_roop import loop_pred


logger = getLogger(__name__)


@hydra.main(config_path="./config", config_name="conll2003", version_base="1.1")
def main(cfg):
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.visible_devices
    os.environ["TRANSFORMERS_CACHE"] = cfg.huggingface_cache

    logger.info(f"Train Roberta\n sub_reg_p={cfg.p}\nseed={cfg.seed}")
    train(
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        num_epoch=cfg.num_epoch,
        length=cfg.length,
        p=cfg.p,
        seed=cfg.seed,
        model_name=cfg.model_name,
        accum_iter=cfg.accum_iter,
        weight_decay=cfg.weight_decay,
        use_loss_weight=cfg.use_loss_weight,
        use_scheduler=cfg.use_scheduler,
        warmup_late=cfg.warmup_late,
        post_sentence_padding=cfg.post_sentence_padding,
        add_sep_between_sentences=cfg.add_sep_between_sentences,
    )
    local_model = "./model/epoch19.pth"
    logger.info("Predict 2023 data")
    output_path = "./many_pred_test.txt"
    loop_pred(
        length=cfg.length,
        model_name=cfg.model_name,
        test=2023,
        loop=cfg.loop,
        batch_size=cfg.test_batch,
        p=cfg.pred_p,
        vote=cfg.vote,
        local_model=local_model,
        post_sentence_padding=cfg.post_sentence_padding,
        add_sep_between_sentences=cfg.add_sep_between_sentences,
        device=cfg.device,
        output_path=output_path,
    )
    logger.info("Predict valid data")
    output_path = "./many_pred_valid.txt"
    loop_pred(
        length=cfg.length,
        model_name=cfg.model_name,
        test="valid",
        loop=cfg.loop,
        batch_size=cfg.test_batch,
        p=cfg.pred_p,
        vote=cfg.vote,
        local_model=local_model,
        post_sentence_padding=cfg.post_sentence_padding,
        add_sep_between_sentences=cfg.add_sep_between_sentences,
        device=cfg.device,
        output_path=output_path,
    )
    logger.info("Predict 2023 data")
    output_path = "./many_pred_2023.txt"
    loop_pred(
        length=cfg.length,
        model_name=cfg.model_name,
        test="2003",
        loop=cfg.loop,
        batch_size=cfg.test_batch,
        p=cfg.pred_p,
        vote=cfg.vote,
        local_model=local_model,
        post_sentence_padding=cfg.post_sentence_padding,
        add_sep_between_sentences=cfg.add_sep_between_sentences,
        device=cfg.device,
        output_path=output_path,
    )

    final_time = time.time() - start
    hours = final_time // 3600
    minutes = final_time // 60 - hours * 60
    seconds = final_time % 60
    logger.info(f"Time: {hours}h {minutes}m {seconds}s")
