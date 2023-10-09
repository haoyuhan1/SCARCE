# from emp import EMP
# from emp_both import EMP
# from prop_mf import MFProp
from prop import Propagation

def get_model(args, dataset, all_features):
    if 'adv' in args.dataset:
        data = dataset.data
    else:
        data = dataset[0]
    data.all_features = all_features
    print('data feature', data.all_features)

    # if args.ogb:
    #     from model_ogb import SAGE, GCN, APPNP, MLP
    #     # from egnn import ElasticGNN_OGB as ElasticGNN
    # else:
    from model import SAGE, GCN, APPNP, MLP, SGC, GAT, APPNP_Hidden, APPNP_Concat, IAPPNP, CSMLP, ORTGNN, LP, SIGN #, ChebNet
        # from egnn import ElasticGNN
    if args.model == 'SAGE':
        model = SAGE(in_channels=data.all_features,
                     hidden_channels=args.hidden_channels,
                     out_channels=dataset.num_classes, 
                     dropout=args.dropout,
                     num_layers=args.num_layers)

    elif args.model == 'GCN':
        model =  GCN(in_channels=data.all_features,
                     hidden_channels=args.hidden_channels,
                     out_channels=dataset.num_classes, 
                     dropout=args.dropout,
                     num_layers=args.num_layers)

    # elif args.model == 'Cheb':
    #     model = ChebNet(in_channels=data.num_features, 
    #                     hidden_channels=args.hidden_channels, 
    #                     out_channels=dataset.num_classes, 
    #                     dropout=args.dropout).cuda()

    elif args.model == 'SGC':
        model = SGC(in_channels=data.all_features,
                    out_channels=dataset.num_classes, 
                    dropout=args.dropout)

    elif args.model == 'GAT':
        # model = GAT(in_channels=data.all_features,
        #             hidden_channels=32,
        #             num_layers=args.num_layers,
        #             heads=8,
        #             out_channels=dataset.num_classes,
        #             dropout=args.dropout).cuda()
        model = GAT(in_channels=data.all_features,
                    hidden_channels=args.hidden_channels,
                    num_layers=args.num_layers,
                    heads=1,
                    out_channels=dataset.num_classes,
                    dropout=args.dropout)

    elif args.model == 'MLP':
        prop = Propagation(K=args.K,
                           alpha=args.alpha,
                           mode='APPNP',
                           cached=True,
                           args=args)

        model =  MLP(in_channels=data.all_features,
                     hidden_channels=args.hidden_channels,
                     out_channels=dataset.num_classes, 
                     dropout=args.dropout,
                     args=args,
                     prop=prop)

    elif args.model == 'SIGN':
        model = SIGN(in_channels=data.all_features,
                     hidden_channels=args.hidden_channels,
                     out_channels=dataset.num_classes, 
                     dropout=args.dropout,
                     args=args,
                     )

    elif args.model == 'APPNP':
        # prop =  EMP(K=args.K,
        #             alpha=args.alpha,
        #             mode='APPNP',
        #             cached=True,
        #             args=args)
        prop = Propagation(K=10,
                           alpha=args.alpha,
                           mode='APPNP',
                           cached=True,
                           args=args)

        model = APPNP(in_channels=data.all_features,
                      hidden_channels=args.hidden_channels, 
                      out_channels=dataset.num_classes, 
                      dropout=args.dropout,
                      num_layers=args.num_layers, 
                      prop=prop,
                      args=args)

    elif args.model == 'IAPPNP':
        # prop =  EMP(K=args.K,
        #             alpha=args.alpha,
        #             mode='APPNP',
        #             cached=True,
        #             args=args)
        prop = Propagation(K=args.K,
                           alpha=args.alpha,
                           mode=args.prop,
                           cached=True,
                           args=args)
        model = IAPPNP(in_channels=data.all_features,
                      hidden_channels=args.hidden_channels,
                      out_channels=dataset.num_classes,
                      dropout=args.dropout,
                      num_layers=args.num_layers,
                      prop=prop,
                      args=args)

    elif args.model == 'ORTGNN':
        model = ORTGNN(in_channels=data.all_features,
                      hidden_channels=args.hidden_channels,
                      out_channels=dataset.num_classes,
                      dropout=args.dropout,
                      num_layers=args.num_layers,
                      args=args)

    elif args.model == 'ElasticGNN':
        if args.prop is None: args.prop = 'EMP' 
        prop =  EMP(K=args.K, 
                    lambda1=args.lambda1,
                    lambda2=args.lambda2,
                    L21=args.L21,
                    alpha=args.alpha, 
                    mode=args.prop, 
                    cached=True, 
                    args=args)

        model = ElasticGNN(in_channels=data.all_features,
                           hidden_channels=args.hidden_channels, 
                           out_channels=dataset.num_classes, 
                           dropout=args.dropout, 
                           num_layers=args.num_layers, 
                           prop=prop).cuda()

    elif args.model == 'MFGNN':
        if args.prop is None: args.prop = 'MFProp' 
        prop =  MFProp(K=args.K, 
                    lambda1=args.lambda1,
                    lambda2=args.lambda2,
                    L21=args.L21,
                    alpha=args.alpha, 
                    mode=args.prop, 
                    cached=True, 
                    args=args)

        model = ElasticGNN(in_channels=data.all_features,
                           hidden_channels=args.hidden_channels, 
                           out_channels=dataset.num_classes, 
                           dropout=args.dropout, 
                           num_layers=args.num_layers, 
                           prop=prop).cuda()
   
    elif args.model == 'MFGNN-Hidden':
        if args.prop is None: args.prop = 'MFProp' 
        prop =  MFProp(K=args.K, 
                    lambda1=args.lambda1,
                    lambda2=args.lambda2,
                    gamma=args.gamma,
                    L21=args.L21,
                    alpha=args.alpha, 
                    mode=args.prop, 
                    cached=True, 
                    args=args)

        model = APPNP_Hidden(in_channels=data.all_features,
                           hidden_channels=args.hidden_channels, 
                           out_channels=dataset.num_classes, 
                           dropout=args.dropout, 
                           num_layers=args.num_layers, 
                           prop=prop).cuda()

    elif args.model == 'MFGNN_Concat':
        if args.prop is None: args.prop = 'MFProp' 
        prop =  MFProp(K=args.K, 
                    lambda1=args.lambda1,
                    lambda2=args.lambda2,
                    gamma=args.gamma,
                    L21=args.L21,
                    alpha=args.alpha, 
                    mode=args.prop, 
                    cached=True, 
                    args=args)

        model = APPNP_Concat(in_channels=data.all_features,
                           hidden_channels=args.hidden_channels, 
                           out_channels=dataset.num_classes, 
                           dropout=args.dropout, 
                           num_layers=args.num_layers, 
                           prop=prop,
                           dataset=dataset).cuda()

    elif args.model == 'ALTOPT':
        prop =  Propagation(K=args.K, 
                    alpha=args.alpha, 
                    mode=args.prop,
                    cached=True,
                    args=args)
        
        model = ALTOPT(in_channels=data.all_features,
                       hidden_channels=args.hidden_channels, 
                       out_channels=dataset.num_classes, 
                       dropout=args.dropout, 
                       num_layers=args.num_layers, 
                       prop=prop,
                       args=args).cuda()
    elif args.model == 'APPNPALT':
        prop = Propagation(K=args.K,
                           alpha=args.alpha,
                           mode=args.prop,
                           cached=True,
                           args=args)

        model = APPNP_ALT(in_channels=data.all_features,
                      hidden_channels=args.hidden_channels,
                      out_channels=dataset.num_classes,
                      dropout=args.dropout,
                      num_layers=args.num_layers,
                      prop=prop,
                      args=args).cuda()
    elif args.model == 'CS':
        model = CSMLP(in_channels=data.all_features,
                    hidden_channels=args.hidden_channels,
                    out_channels=dataset.num_classes,
                    dropout=args.dropout,
                    args=args).cuda()
        prop = Propagation(K=args.K,
                           alpha=args.alpha,
                           mode=args.prop,
                           cached=True,
                           args=args)
        # model = IAPPNP(in_channels=data.all_features,
        #                hidden_channels=args.hidden_channels,
        #                out_channels=dataset.num_classes,
        #                dropout=args.dropout,
        #                num_layers=args.num_layers,
        #                prop=prop,
        #                args=args).cuda()
    elif args.model == 'LP':
        model = LP(args=args)

    else:
        raise ValueError('Model not supported')

    return model