import torch.optim as optim

if __name__=='__main__':
    # basic config
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default='test')
    parser.add_argument('--checkpoints',      type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--optimizer_type',   type=str, default='sgd', choices=['sgd', 'sam'])
    parser.add_argument('--rho',              type=float, default=0.1)
    parser.add_argument('--redo',             type=str2bool, default=False)
    parser.add_argument('--redo_tau',         type=float, default=0.1)
    parser.add_argument('--backbone_reset',   type=str2bool, default=False)
    parser.add_argument('--policy_reset',     type=str2bool, default=False)
    parser.add_argument('--backbone_norm',    type=str, default=None)
    parser.add_argument('--policy_norm',      type=str, default=None)
    parser.add_argument('--backbone_crelu',   type=str2bool, default=False)
    parser.add_argument('--policy_crelu',     type=str2bool, default=False)
    parser.add_argument('--weight_decay',     type=float, default=0)
    parser.add_argument('--batch_size',       type=int, default=128)
    parser.add_argument('--learning_rate',    type=float, default=0.001)
    parser.add_argument('--momentum',         type=float, default=0.9)
    parser.add_argument('--gpu',              type=int, default=0)
    parser.add_argument('--num_iters',        type=int, default=50000)
    parser.add_argument('--num_chunks',       type=int, default=100)
    parser.add_argument('--num_workers',   type=int, default=0)
    parser.add_argument('--seed',          type=int, default=2021)

    args = parser.parse_args()
    run(vars(args))

optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay, 
            momentum=args.momentum,
            rho=args.rho
        )