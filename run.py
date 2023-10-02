import hydra

import bert_pred
import bert_train


@hydra.main(config_path="./config", config_name="conll2003", version_base="1.1")
def main(cfg):
    batch_size = cfg.batch_size
    lr = cfg.lr
    num_epoch = cfg.num_epoch
    warmup_t = cfg.warmup_t
    length = cfg.length
    p = cfg.p
    seed = cfg.seed
    device = cfg.device
    model_name = cfg.model_name
    use_scheduler = cfg.use_scheduler
    stop_word = cfg.stop_word
    as_aug = cfg.as_aug
    path = f"./model/epoch{num_epoch-1}.pth"

    lr_min = cfg.lr_min
    warmup_lr_init = cfg.warmup_lr_init
    boi1 = cfg.boi1

    test = cfg.test
    leak = cfg.leak

    bert_train.train(
        batch_size,
        lr,
        num_epoch,
        warmup_t,
        length,
        p,
        seed,
        device,
        model_name,
        lr_min,
        warmup_lr_init,
        use_scheduler,
        stop_word,
        as_aug,
        boi1,
    )
    bert_pred.pred(length, path, test, leak)


if __name__ == "__main__":
    main()
