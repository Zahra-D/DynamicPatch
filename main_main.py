
import random 
import numpy as np
import torch
import DynamicTimm
from DynamicTimm.data import Mixup
from DynamicTimm.loss import SoftTargetCrossEntropy
from DynamicTimm.scheduler import create_scheduler
from DynamicTimm.optim import create_optimizer
from DynamicTimm.utils import accuracy

import json
from pathlib import Path
import time
import datetime
import logging
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter



from dataset import get_dataset
from utils import save_checkpoint, load_checkpoint
from args import get_parser



logging.basicConfig(
format="%(asctime)s \n %(message)s\n", level=logging.INFO
)
logger = logging.getLogger(__name__)


def train_one_epoch(model: torch.nn.Module, criterion, mixup_fn, 
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device,writer, epoch: int, args = None):
    
    logger.info('start training')
    model.train()
    


    train_iterator = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", leave=False)
    for batch_idx, batch in enumerate(train_iterator):

        # if batch_idx >= 10:
        #     break

        
        inputs, targets = batch['image'].to(device), batch['label'].to(device)  
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)


        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()


        # Update model parameters
        
        loss = loss.detach().cpu()
        global_step = epoch * len(data_loader) + batch_idx
        writer.add_scalar('Loss/trainG', loss.item(), global_step)
        writer.add_scalar('LR/step', optimizer.param_groups[0]['lr'], global_step)
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         try:
        #             writer.add_histogram(f'{name}.grad', param.grad, global_step)
        #         except Exception as e:
        #             print(e)
        # print(epoch, batch_idx)
        optimizer.step()
        train_iterator.set_postfix(loss=loss.item())
        

    writer.add_scalar('LR/Epoch', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('Loss/trainE', loss.item(), epoch)


def evaluate(dataloader_test, model, writer, epoch, device):
    logger.info('start evaluatiing')
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    acc_top1 = 0
    acc_top5 = 0
    bs_s = 0
    with torch.no_grad():
        
        iterator = tqdm(dataloader_test, desc=f"validation", unit="batch", leave=False)
        for batch_idx, batch in enumerate(iterator):
            # Move inputs and targets to device if using GPU
            inputs, targets = batch['image'].to(device), batch['label'].to(device)


            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            bs = len(targets)
            
            acc_top1 += acc1 * bs
            acc_top5 += acc5 * bs
            bs_s += bs
            

        writer.add_scalar('Loss/val', loss.detach().cpu().item(), epoch)
        writer.add_scalar('Accuracy/val_top1', acc_top1/bs_s, epoch)
        writer.add_scalar('Accuracy/val_top5', acc_top5/bs_s, epoch)
        
    

    

def main(args):

    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    
    
    


    
    

    dataset_train = get_dataset(args, type='train')
    dataset_val = get_dataset(args, type='validation')
    
    
    
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
   

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    

    
    if mixup_active:
        logger.info('we are using mixup')
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    

    
    

    
    
    output_dir = Path(args.output_dir)
    writer = SummaryWriter(output_dir / Path(f'runs'))
    
    args_dict = vars(args)
    with open(output_dir/ Path('arguments.json'), 'w') as file:
        json.dump(args_dict, file, indent=4)
        
        
        
        
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
    if args.resume:
        model = DynamicTimm.create_model(args.model, pretrained=True)
        model.to(device)
        optimizer = create_optimizer(args, model)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        checkpoint_path = output_dir/ Path(f'epoch{args.start_epoch-1}/check_point.pt')
        load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path)
    if not args.resume:
        # args.start_epoch = 0
        model = DynamicTimm.create_model(args.model)
        model.to(device)
        optimizer = create_optimizer(args, model)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        
        
    
    model.train()
    
    
    print('device is ',device )
    
    
     
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        
        print(f'epoch number {epoch}')

        train_one_epoch(
            model, criterion,mixup_fn, data_loader_train,
            optimizer, device,writer, epoch,
            args = args,
        )
        lr_scheduler.step(epoch)
        
        if ((epoch+1)%2) == 0:
            save_checkpoint(model,optimizer, lr_scheduler, output_dir, epoch, args)
             

        evaluate(data_loader_val, model, writer, epoch, device)
        # print(f"Accuracy of the network on the ~5000 test images: {test_stats['acc1']:.1f}%") 
            
        # print(f'Max accuracy: {max_accuracy:.2f}%')


        
        
        
        


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # torch.save(model, f'{output_dir}/trained_model_learnablePosEmbed.pt')

    
    
    
    
if __name__ == '__main__':
    
    
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    parser = get_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)