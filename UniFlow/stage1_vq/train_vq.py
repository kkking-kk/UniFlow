import os
import sys
import hashlib
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import options.option_vq as option_vq
args = option_vq.get_args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae

import utils.utils_model as utils_model
from dataset import dataset_VQ
import utils.eval_uniflow as eval_trans

import warnings
warnings.filterwarnings('ignore')
from torch.autograd import Variable
CUDA_LAUNCH_BLOCKING=1


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


def _count_parameters(module):
    return sum(param.numel() for param in module.parameters())


def _format_list(values):
    if values is None:
        return "None"
    return "[" + ", ".join(str(value) for value in values) + "]"


def resolve_tensorboard_logdir(out_dir):
    if os.name != "nt" and all(ord(ch) < 128 for ch in out_dir):
        return out_dir, False

    digest = hashlib.md5(out_dir.encode("utf-8")).hexdigest()[:10]
    exp_name = os.path.basename(out_dir.rstrip("\\/")) or "default"
    safe_exp_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in exp_name)
    tb_root = os.path.join(tempfile.gettempdir(), "uniflow_tensorboard")
    tb_dir = os.path.join(tb_root, f"{safe_exp_name}_{digest}")
    os.makedirs(tb_dir, exist_ok=True)

    pointer_path = os.path.join(out_dir, "tensorboard_logdir.txt")
    with open(pointer_path, "w", encoding="utf-8") as fp:
        fp.write(tb_dir + "\n")

    return tb_dir, True


def log_stage1_model_summary(logger, net, sample_batch):
    vq_module = net.vqvae
    if getattr(vq_module, "use_multiscale", False):
        model_type = f"multi-scale ({getattr(vq_module, 'multiscale_mode', 'unknown')})"
    else:
        model_type = "single-scale"
    logger.info(f"Stage1 tokenizer mode: {model_type}")
    logger.info(f"Total parameters: {_count_parameters(net):,}")
    logger.info(f"Sample batch shape for summary: {tuple(sample_batch.shape)}")

    with torch.no_grad():
        x_in = vq_module.preprocess(sample_batch[:1].float())
        if getattr(vq_module, "use_multiscale", False):
            branch_outputs = [encoder(x_in) for encoder in vq_module.encoders]
            seq_lens = [tensor.shape[-1] for tensor in branch_outputs]
            branch_channels = [tensor.shape[1] for tensor in branch_outputs]
            scale_dims = list(getattr(vq_module, "scale_dims", branch_channels))
            scale_depths = list(getattr(vq_module, "scale_depths", []))
            scale_dilations = list(getattr(vq_module, "scale_dilations", []))
            scale_weights = getattr(vq_module.quantizer, "scale_weights", None)
            scale_patches = getattr(vq_module, "scale_patches", None)
            scale_patch_downs = getattr(vq_module, "scale_patch_downs", None)
            scale_raw_lengths = getattr(vq_module, "scale_raw_lengths", None)
            scale_aligned_lengths = getattr(vq_module, "scale_aligned_lengths", None)
            decoder_input_length = getattr(vq_module, "decoder_input_length", seq_lens[0])

            logger.info("Multi-scale encoder summary:")
            logger.info(f"  Multi-scale mode: {getattr(vq_module, 'multiscale_mode', 'unknown')}")
            logger.info(f"  Branch count: {len(branch_outputs)}")
            logger.info(f"  Scale dims: {_format_list(scale_dims)}")
            logger.info(f"  Scale depths: {_format_list(scale_depths)}")
            logger.info(f"  Scale dilations: {_format_list(scale_dilations)}")
            logger.info(f"  Shared-mapping weights: {_format_list(scale_weights)}")
            if scale_patches is not None:
                logger.info(f"  Patch sizes: {_format_list(scale_patches)}")
                logger.info(f"  Patch downsample stages: {_format_list(scale_patch_downs)}")
                logger.info(f"  Raw patch token lengths: {_format_list(scale_raw_lengths)}")
                logger.info(f"  Aligned branch token lengths: {_format_list(scale_aligned_lengths)}")
            logger.info(f"  Branch token lengths: {_format_list(seq_lens)}")
            logger.info(f"  Branch latent channels: {_format_list(branch_channels)}")
            logger.info(
                f"  Shared index sequence length: {seq_lens[0]} | "
                f"Public fused codebook shape: {tuple(vq_module.quantizer.codebook.shape)} | "
                f"Decoder input token length: {decoder_input_length}"
            )

            for idx, encoder in enumerate(vq_module.encoders):
                codebook_shape = tuple(vq_module.quantizer.scale_quantizers[idx].codebook.shape)
                logger.info(
                    f"  Branch {idx}: params={_count_parameters(encoder):,}, "
                    f"latent_shape={tuple(branch_outputs[idx].shape)}, "
                    f"codebook_shape={codebook_shape}"
                )
            logger.info(
                f"  Decoder params={_count_parameters(vq_module.decoder):,}, "
                f"decoder_input_dim={vq_module.code_dim}"
            )
        else:
            x_encoder = vq_module.encoder(x_in)
            logger.info("Single-scale encoder summary:")
            logger.info(
                f"  Encoder params={_count_parameters(vq_module.encoder):,}, "
                f"latent_shape={tuple(x_encoder.shape)}"
            )
            logger.info(
                f"  Quantizer codebook shape={tuple(vq_module.quantizer.codebook.shape)}, "
                f"shared_index_sequence_length={x_encoder.shape[-1]}"
            )
            logger.info(
                f"  Decoder params={_count_parameters(vq_module.decoder):,}, "
                f"decoder_input_dim={vq_module.code_dim}"
            )


##### ---- Exp dirs ---- #####
# args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.out_dir = os.path.abspath(args.out_dir)
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
tb_logdir, tb_is_fallback = resolve_tensorboard_logdir(args.out_dir)
writer = SummaryWriter(tb_logdir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
if tb_is_fallback:
    logger.info(f"TensorBoard log dir redirected to ASCII-safe path: {tb_logdir}")
else:
    logger.info(f"TensorBoard log dir: {tb_logdir}")


logger.info(f'Training on {args.dataname}')
logger.info(f"Dataloader workers: {args.num_workers}")




##### ---- Dataloader ---- #####
train_loader = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        num_workers=args.num_workers,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t, dataset_type='train')

train_loader_iter = dataset_VQ.cycle(train_loader)
sample_batch = next(iter(train_loader))


MSELoss = torch.nn.MSELoss()
##### ---- Network ---- #####
net = vqvae.VQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)


if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
log_stage1_model_summary(logger, net, sample_batch)
net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
  


##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in range(1, args.warm_up_iter):

    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    input = next(train_loader_iter)
    input = input.cuda().float()
    output, loss_commit, perplexity = net(input)
    loss_recons = MSELoss(output, input)
    loss = args.recon*loss_recons + args.commit * loss_commit

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_recons.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()

    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter

        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.8f}")

        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

####---- Training ---- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
best_iter, best_ds, best_mse, writer, logger = eval_trans.evaluation_vqvae(args,args.out_dir, train_loader, net, logger, writer, 0, best_iter=0, best_ds=99999, best_mse = 99999)

for nb_iter in range(1, args.total_iter + 1):
    
    input = next(train_loader_iter)
    input = input.cuda().float() # bs, nb_joints, joints_dim, seq_len
    
    output, loss_commit, perplexity = net(input)

    loss_recons = MSELoss(output, input)
    loss = args.recon*loss_recons + args.commit * loss_commit


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_recons.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter == 0:
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        writer.add_scalar('./Train/Recons', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        
        logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.8f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.8f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

    if nb_iter % args.eval_iter==0:
        best_iter, best_ds, best_mse, writer, logger = eval_trans.evaluation_vqvae(args,args.out_dir, train_loader, net, logger, writer, nb_iter, best_iter, best_ds, best_mse)
msg_final = f"Train. Iter {best_iter} : , MAE. {best_ds:.4f}, MSE. {best_mse:.6f}"
logger.info(msg_final)
