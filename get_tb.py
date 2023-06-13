import os


tgt_dir = "./tensorboard"
src_dir = "./output"
model_list = ["newgpt-medium-scratch-default-debug-0.0003-full", "bloom-medium-scratch-default-debug-0.0003-full", "newgpt-medium-scratch-rot-decay-fix-0.0003-full"]


os.system("rm -r {}/*".format(tgt_dir))
for model in model_list:
    src_model_dir = os.path.join(src_dir, model, "tb-logs/train_inner")
    tgt_model_dir = os.path.join(tgt_dir, model)
    os.mkdir(tgt_model_dir)
    os.system("cp {}/* {}".format(src_model_dir, tgt_model_dir))

