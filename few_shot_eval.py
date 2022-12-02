import torch
import numpy as np
from args import *
from utils import *

n_runs = args.n_runs
batch_few_shot_runs = 1
assert(n_runs % batch_few_shot_runs == 0)

def define_runs(n_ways, n_shots, n_queries, num_classes, elements_per_class):
    shuffle_classes = torch.LongTensor(np.arange(num_classes))
    run_classes = torch.LongTensor(n_runs, n_ways).to(args.device)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shots + n_queries).to(args.device)
    for i in range(n_runs):
        run_classes[i] = torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            # print(n_runs,n_ways,n_queries,run_classes[i])
            # print(run_classes,run_classes[i, j],elements_per_class[run_classes[i, j]])
            # while 1:
            #     pass
            run_indices[i,j] = torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices

def generate_runs(data, run_classes, run_indices, batch_idx):
    n_runs, n_ways, n_samples = run_classes.shape[0], run_classes.shape[1], run_indices.shape[2]
    run_classes = run_classes[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    run_indices = run_indices[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    run_classes = run_classes.unsqueeze(2).unsqueeze(3).repeat(1,1,data.shape[1], data.shape[2])
    run_indices = run_indices.unsqueeze(3).repeat(1, 1, 1, data.shape[2])
    datas = data.unsqueeze(0).repeat(batch_few_shot_runs, 1, 1, 1)
    cclasses = torch.gather(datas, 1, run_classes)
    res = torch.gather(cclasses, 2, run_indices)
    return res

def ncm(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def transductive_ncm(train_features, features, run_classes, run_indices, n_shots, n_iter_trans = args.transductive_n_iter, n_iter_trans_sinkhorn = args.transductive_n_iter_sinkhorn, temp_trans = args.transductive_temperature, alpha_trans = args.transductive_alpha, cosine = args.transductive_cosine, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        if cosine:
            features = features / torch.norm(features, dim = 2, keepdim = True)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            if cosine:
                means = means / torch.norm(means, dim = 2, keepdim = True)
            for _ in range(n_iter_trans):
                if cosine:
                    similarities = torch.einsum("bswd,bswd->bsw", runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim), means.reshape(batch_few_shot_runs, 1, args.n_ways, dim))
                    soft_sims = torch.softmax(temp_trans * similarities, dim = 2)
                else:
                    similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                    soft_sims = torch.exp( -1 * temp_trans * similarities)
                for _ in range(n_iter_trans_sinkhorn):
                    soft_sims = soft_sims / soft_sims.sum(dim = 2, keepdim = True) * args.n_ways
                    soft_sims = soft_sims / soft_sims.sum(dim = 1, keepdim = True) * args.n_queries
                new_means = ((runs[:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", soft_sims, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3])))) / runs.shape[2]
                if cosine:
                    new_means = new_means / torch.norm(new_means, dim = 2, keepdim = True)
                means = means * alpha_trans + (1 - alpha_trans) * new_means
                if cosine:
                    means = means / torch.norm(means, dim = 2, keepdim = True)
            if cosine:
                winners = torch.max(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            else:
                winners = torch.min(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def kmeans(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            for i in range(500):
                similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                new_allocation = (similarities == torch.min(similarities, dim = 2, keepdim = True)[0]).float()
                new_allocation = new_allocation / new_allocation.sum(dim = 1, keepdim = True)
                allocation = new_allocation
                means = (runs[:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", allocation, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3])) * args.n_queries) / runs.shape[2]
            winners = torch.min(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def softkmeans(train_features, features, run_classes, run_indices, n_shots, transductive_temperature_softkmeans=args.transductive_temperature_softkmeans, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            runs = postprocess(runs)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            for i in range(30):
                similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                soft_allocations = F.softmax(-similarities.pow(2)*args.transductive_temperature_softkmeans, dim=2)
                means = torch.sum(runs[:,:,:n_shots], dim = 2) + torch.einsum("rsw,rsd->rwd", soft_allocations, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3]))
                means = means/(n_shots+soft_allocations.sum(dim = 1).reshape(batch_few_shot_runs, -1, 1))
            winners = torch.min(similarities, dim = 2)[1]
            winners = winners.reshape(batch_few_shot_runs, args.n_ways, -1)
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def ncm_cosine(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        features = sphering(features)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            means = sphering(means)
            distances = torch.einsum("bwysd,bwysd->bwys",runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim), means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim))
            winners = torch.max(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def get_features(model, loader, n_aug = args.sample_aug):
    model.eval()
    for augs in range(n_aug):
        all_features, offset, max_offset = [], 1000000, 0
        for batch_idx, (data, target) in enumerate(loader):        
            with torch.no_grad():
                data, target = data.to(args.device), target.to(args.device)
                _, features = model(data)
                all_features.append(features)
                offset = min(min(target), offset)
                max_offset = max(max(target), max_offset)
        num_classes = max_offset - offset + 1
        print(".", end='')
        if augs == 0:
            features_total = torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
        else:
            features_total += torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
    return features_total / n_aug

def get_features2(model, loader, n_aug = args.sample_aug):
    model.eval()
    for augs in range(n_aug):
        all_features, offset, max_offset = [], 1000000, 0
        for batch_idx, (data, target) in enumerate(loader):
            with torch.no_grad():
                data, target = data.to(args.device), target.to(args.device)
                _, features = model(data)
                all_features.append(features)
                offset = min(min(target), offset)
                max_offset = max(max(target), max_offset)
        num_classes = max_offset - offset + 1
        print(".", end='')
        if augs == 0:
            features_total = torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
        else:
            features_total = torch.cat([features_total,torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])],dim = 2)
    return features_total,num_classes

def eval_few_shot(train_features, val_features, novel_features, val_run_classes, val_run_indices, novel_run_classes, novel_run_indices, n_shots, transductive = False,elements_train=None):
    if transductive:
        if args.transductive_softkmeans:
            return softkmeans(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), softkmeans(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
        else:
            return kmeans(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), kmeans(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
    else:
        return ncm(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), ncm(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)

def update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data):

    if "M" in args.preprocessing or args.save_features != '':
        train_features = get_features(model, train_clean)
    else:
        train_features = torch.Tensor(0,0,0)
    val_features = get_features(model, val_loader)
    novel_features = get_features(model, novel_loader)
    ###
    # if "M" in args.preprocessing or args.save_features != '':
    #     train_features,num_classes_train = get_features2(model, train_clean)
    # else:
    #     train_features = torch.Tensor(0,0,0)
    # val_features,num_classes_val = get_features2(model, val_loader)
    #
    # novel_features,num_classes = get_features2(model, novel_loader)
    # if args.save_features != "":
    #     torch.save(torch.cat([train_features, val_features, novel_features], dim=0),
    #                args.save_features + str(args.n_shots[0]))
    # novel_features=torch.sum(novel_features.view(num_classes, -1, novel_features.shape[2]//args.sample_aug,args.sample_aug),dim=3)//args.sample_aug
    # train_features = torch.mean(
    #     train_features.view(num_classes_train, -1, train_features.shape[2] // args.sample_aug, args.sample_aug), dim=-1)
    # val_features = torch.mean(
    #     val_features.view(num_classes_val, -1, val_features.shape[2] // args.sample_aug, args.sample_aug), dim=-1)
    ###
    res = []
    for i in range(len(args.n_shots)):
        res.append(evaluate_shot(i, train_features, val_features, novel_features, few_shot_meta_data, model = model))

    return res

def evaluate_shotProduFeature(index, train_features, val_features, novel_features, few_shot_meta_data, model = None, transductive = False):
    (val_acc, val_conf), (novel_acc, novel_conf) = eval_few_shot(train_features, val_features, novel_features, few_shot_meta_data["val_run_classes"][index], few_shot_meta_data["val_run_indices"][index], few_shot_meta_data["novel_run_classes"][index], few_shot_meta_data["novel_run_indices"][index], args.n_shots[index], transductive = transductive, elements_train=few_shot_meta_data["elements_train"])
    if val_acc > few_shot_meta_data["best_val_acc"][index]:
        if val_acc > few_shot_meta_data["best_val_acc_ever"][index]:
            few_shot_meta_data["best_val_acc_ever"][index] = val_acc
            if args.save_model != "":
                if len(args.devices) == 1:
                    torch.save(model.state_dict(), args.save_model + str(args.n_shots[index]))
                else:
                    torch.save(model.module.state_dict(), args.save_model + str(args.n_shots[index]))
            #if args.save_features != "":
            #    torch.save(torch.cat([train_features, val_features, novel_features], dim = 0), args.save_features + str(args.n_shots[index]))
        few_shot_meta_data["best_val_acc"][index] = val_acc
        few_shot_meta_data["best_novel_acc"][index] = novel_acc
    return val_acc, val_conf, novel_acc, novel_conf



def evaluate_shot(index, train_features, val_features, novel_features, few_shot_meta_data, model = None, transductive = False):
    (val_acc, val_conf), (novel_acc, novel_conf) = eval_few_shot(train_features, val_features, novel_features, few_shot_meta_data["val_run_classes"][index], few_shot_meta_data["val_run_indices"][index], few_shot_meta_data["novel_run_classes"][index], few_shot_meta_data["novel_run_indices"][index], args.n_shots[index], transductive = transductive, elements_train=few_shot_meta_data["elements_train"])
    if val_acc > few_shot_meta_data["best_val_acc"][index]:
        if val_acc > few_shot_meta_data["best_val_acc_ever"][index]:
            few_shot_meta_data["best_val_acc_ever"][index] = val_acc
            if args.save_model != "":
                if len(args.devices) == 1:
                    torch.save(model.state_dict(), args.save_model + str(args.n_shots[index]))
                else:
                    torch.save(model.module.state_dict(), args.save_model + str(args.n_shots[index]))
            if args.save_features != "":
                torch.save(torch.cat([train_features, val_features, novel_features], dim = 0), args.save_features + str(args.n_shots[index]))
        few_shot_meta_data["best_val_acc"][index] = val_acc
        few_shot_meta_data["best_novel_acc"][index] = novel_acc
    return val_acc, val_conf, novel_acc, novel_conf

from s2m2 import distLinear
from torch.autograd import Variable
def evaluate_shotMutiFeature(index, train_features, val_features, novel_features, few_shot_meta_data, model = None, transductive = False):
    args.preprocessing='ME'
    sample_num = 5
    # # print(train_features.shape)
    # print(novel_features.shape)
    # train_features=torch.mean(train_features.view(len(train_features), -1, sample_num,train_features.shape[2]//sample_num),dim=2)
    # novel_features=novel_features.view(len(novel_features), -1, novel_features.shape[2]//sample_num)
    novel_features = preprocess(train_features, novel_features, elements_train=few_shot_meta_data["elements_train"])
    acc_all1, acc_all2, acc_all3 = [], [], []

    for cot,i in enumerate(few_shot_meta_data["novel_run_classes"][index]):

        # z_support = novel_features[i,:n_shots*sample_num]
        # z_query = torch.mean(novel_features[i,n_shots*sample_num:].view(len(i), -1,sample_num, novel_features.shape[2]),dim=2)
        z_support=novel_features[i,:n_shots]
        z_query = novel_features[i, n_shots :]
        n_support =z_support.shape[1]
        n_query=z_query.shape[1]
        n_ways=len(i)


        z_support   = z_support.view(n_support*n_ways, -1 ).to(args.device)
        z_query     = z_query.view(n_query*n_ways, -1 ).to(args.device)

        y_support = torch.from_numpy(np.repeat(range( n_ways ), n_support )).to(args.device)
        #print(y_support.shape)
        #y_support = y_support.to(args.device)

        y_query = torch.from_numpy(np.repeat(range(n_ways), n_query)).to(args.device)

        #y_query = Variable(y_query.cuda())

        linear_clf = distLinear(z_support.shape[1], n_ways)
        linear_clf = linear_clf.to(args.device)

        #set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        set_optimizer = torch.optim.Adam(linear_clf.parameters(), lr=0.01, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(args.device)

        batch_size = 4#*sample_num
        support_size = len(i) * n_support
        res = []
        for epoch in range(61):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()
            if epoch % 20 == 0 and epoch != 0:
                pred=linear_clf(z_query).argmax(axis=1)
                #print(pred.shape)
                res.append(torch.sum(pred == y_query) * 100/pred.shape[0])


        acc_all1.append(res[0].cpu().numpy())
        acc_all2.append(res[1].cpu().numpy())
        acc_all3.append(res[2].cpu().numpy())
        acc_mean1 = np.mean(acc_all1)
        acc_mean2 = np.mean(acc_all2)
        acc_mean3 = np.mean(acc_all3)
        print(cot,acc_mean1,acc_mean2,acc_mean3)

    acc_std1 = np.std(acc_all1)
    acc_std2 = np.std(acc_all2)
    acc_std3 = np.std(acc_all3)
    iter_num=len(few_shot_meta_data["novel_run_indices"][index])
    print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean1, 1.96 * acc_std1 / np.sqrt(iter_num)))
    print('%d Test Acc at 200= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean2, 1.96 * acc_std2 / np.sqrt(iter_num)))
    print('%d Test Acc at 300= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean3, 1.96 * acc_std3 / np.sqrt(iter_num)))


    return 0, 0, acc_mean2, acc_std2


# def evaluate_shot(index, train_features, val_features, novel_features, few_shot_meta_data, model=None,
#                   transductive=False):
#     args.preprocessing = 'M'
#     novel_features = preprocess(train_features, novel_features, elements_train=few_shot_meta_data["elements_train"])
#     acc_all1, acc_all2, acc_all3 = [], [], []
#     sample_num=5
#
#     for cot, i in enumerate(few_shot_meta_data["novel_run_classes"][index]):
#
#         z_support = novel_features[i, :n_shots].view(len(i), -1, sample_num,novel_features.shape[2]//sample_num)
#         z_query = novel_features[i, n_shots:].view(len(i), -1, sample_num,novel_features.shape[2]//sample_num)
#
#         n_support = z_support.shape[1]
#         n_query = z_query.shape[1]
#         n_ways = len(i)
#
#         # z_support = z_support.view(n_support * n_ways, -1).to(args.device)
#         # z_query = z_query.view(n_query * n_ways, -1).to(args.device)
#
#         y_support = torch.from_numpy(np.repeat(range(n_ways), n_support)).to(args.device)
#         # print(y_support.shape)
#         # y_support = y_support.to(args.device)
#
#         y_query = torch.from_numpy(np.repeat(range(n_ways), n_query)).to(args.device)
#
#         # y_query = Variable(y_query.cuda())
#         feat_dim = z_support.shape[-1]
#         model_num=5
#
#
#         for l in range(model_num):
#             start=l*feat_dim//model_num
#             end=l*feat_dim//model_num+feat_dim//model_num
#             # start=l*feature_dim
#             # end=start+feature_dim
#             support_split = z_support[:, :,:,start:end].reshape(n_ways,n_support ,-1).view(n_support * n_ways, -1).to(args.device)
#
#
#             query_split = z_query[:, :,:,start:end].reshape(n_ways,n_query ,-1).view(n_query * n_ways, -1).to(args.device)
#
#             linear_clf = distLinear(support_split.shape[1], n_ways)
#             linear_clf = linear_clf.to(args.device)
#
#             set_optimizer = torch.optim.Adam(linear_clf.parameters(), lr=0.01, weight_decay=0.001)
#
#             loss_function = nn.CrossEntropyLoss()
#             loss_function = loss_function.to(args.device)
#
#             batch_size = 5
#             support_size = n_ways * n_support
#             res_ = []
#             for epoch in range(61):
#                 rand_id = np.random.permutation(support_size)
#                 for i in range(0, support_size, batch_size):
#                     set_optimizer.zero_grad()
#                     selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
#                     z_batch = support_split[selected_id]
#                     y_batch = y_support[selected_id]
#                     scores = linear_clf(z_batch)
#                     loss = loss_function(scores, y_batch)
#                     loss.backward()
#                     set_optimizer.step()
#                 if epoch % 20 == 0 and epoch != 0:
#                     pred = linear_clf(query_split)
#                     # print(pred.shape)
#                     res_.append(pred.detach().cpu().numpy())
#             res_=np.array(res_)
#             if l==0:
#                 r=res_.copy()
#             else:
#                 r+=res_
#
#
#         res=[np.mean(r_.argmax(axis=1) == y_query.cpu().numpy()) * 100 for r_ in r]
#
#         acc_all1.append(res[0])
#         acc_all2.append(res[1])
#         acc_all3.append(res[2])
#         acc_mean1 = np.mean(acc_all1)
#         acc_mean2 = np.mean(acc_all2)
#         acc_mean3 = np.mean(acc_all3)
#         print(cot, acc_mean1, acc_mean2, acc_mean3)
#
#     acc_std1 = np.std(acc_all1)
#     acc_std2 = np.std(acc_all2)
#     acc_std3 = np.std(acc_all3)
#     iter_num = len(few_shot_meta_data["novel_run_indices"][index])
#     print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean1, 1.96 * acc_std1 / np.sqrt(iter_num)))
#     print('%d Test Acc at 200= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean2, 1.96 * acc_std2 / np.sqrt(iter_num)))
#     print('%d Test Acc at 300= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean3, 1.96 * acc_std3 / np.sqrt(iter_num)))
#
#     return 0, 0, acc_mean2, acc_std2

from s2m2 import distLinear
from torch.autograd import Variable
def evaluate_shot4(index, train_features, val_features, novel_features, few_shot_meta_data, model = None, transductive = False):
    #args.preprocessing='ME'
    #novel_features = preprocess(train_features, novel_features, elements_train=few_shot_meta_data["elements_train"])
    acc_all1, acc_all2, acc_all3 = [], [], []

    for cot in range(n_runs // batch_few_shot_runs):
        runs = generate_runs(novel_features, few_shot_meta_data["novel_run_classes"][index],
                             few_shot_meta_data["novel_run_indices"][index], cot)
        runs = runs.squeeze(0)
        z_support = runs[:, :n_shots]

        z_query = runs[:, n_shots:]

        n_support = z_support.shape[1]
        n_query = z_query.shape[1]
        n_ways = z_query.shape[0]

        z_support   = z_support.contiguous().view(n_support*n_ways, -1 ).to(args.device)
        z_query     = z_query.contiguous().view(n_query*n_ways, -1 ).to(args.device)

        y_support = torch.from_numpy(np.repeat(range( n_ways ), n_support )).to(args.device)
        #print(y_support.shape)
        #y_support = y_support.to(args.device)

        y_query = torch.from_numpy(np.repeat(range(n_ways), n_query)).to(args.device)

        #y_query = Variable(y_query.cuda())

        linear_clf = distLinear(z_support.shape[1], n_ways)
        linear_clf = linear_clf.to(args.device)

        #set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        set_optimizer = torch.optim.Adam(linear_clf.parameters(), lr=0.01,
                                        weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(args.device)

        batch_size = 4
        support_size = n_ways * n_support
        res = []
        for epoch in range(61):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()
            if epoch % 20 == 0 and epoch != 0:
                pred=linear_clf(z_query).argmax(axis=1)
                #print(pred.shape)
                res.append(torch.sum(pred == y_query) * 100/pred.shape[0])


        acc_all1.append(res[0].cpu().numpy())
        acc_all2.append(res[1].cpu().numpy())
        acc_all3.append(res[2].cpu().numpy())
        acc_mean1 = np.mean(acc_all1)
        acc_mean2 = np.mean(acc_all2)
        acc_mean3 = np.mean(acc_all3)
        print(cot,acc_mean1,acc_mean2,acc_mean3,1.96 * np.std(acc_all2) / np.sqrt(cot+1))

    acc_std1 = np.std(acc_all1)
    acc_std2 = np.std(acc_all2)
    acc_std3 = np.std(acc_all3)
    iter_num=len(few_shot_meta_data["novel_run_indices"][index])
    print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean1, 1.96 * acc_std1 / np.sqrt(iter_num)))
    print('%d Test Acc at 200= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean2, 1.96 * acc_std2 / np.sqrt(iter_num)))
    print('%d Test Acc at 300= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean3, 1.96 * acc_std3 / np.sqrt(iter_num)))


    return 0, 0, acc_mean2, acc_std2

def evaluate_shotdrop(index, train_features, val_features, novel_features, few_shot_meta_data, model=None,
                  transductive=False):
    args.preprocessing = 'M'
    novel_features = preprocess(train_features, novel_features, elements_train=few_shot_meta_data["elements_train"])
    acc_all1, acc_all2, acc_all3 = [], [], []

    for cot, i in enumerate(few_shot_meta_data["novel_run_classes"][index]):

        z_support = novel_features[i, :n_shots]
        z_query = novel_features[i, n_shots:]

        n_support = z_support.shape[1]
        n_query = z_query.shape[1]
        n_ways = len(i)

        z_support = z_support.view(n_support * n_ways, -1).to(args.device)
        z_query = z_query.view(n_query * n_ways, -1).to(args.device)

        y_support = torch.from_numpy(np.repeat(range(n_ways), n_support)).to(args.device)
        # print(y_support.shape)
        # y_support = y_support.to(args.device)

        y_query = torch.from_numpy(np.repeat(range(n_ways), n_query)).to(args.device)

        # y_query = Variable(y_query.cuda())
        feat_dim = z_support.shape[1]
        drop_rate = 0.2
        feature_dim = int(drop_rate * feat_dim)
        model_num=5


        import random
        select_dim = range(feat_dim)


        support_split =z_support
        query_split = z_query

        linear_clf = distLinear(support_split.shape[1], n_ways)
        linear_clf = linear_clf.to(args.device)

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)
        # set_optimizer = torch.optim.Adam(linear_clf.parameters(), lr=0.01,
        #                                  weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.to(args.device)

        batch_size = 4
        support_size = n_ways * n_support
        res_ = []
        for epoch in range(301):
            rand_id = np.random.permutation(support_size)
            select_id = random.sample(select_dim, feature_dim)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = support_split[selected_id].clone()
                with torch.no_grad():
                    z_batch[:, select_id] = 0
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()
            if epoch % 100 == 0 and epoch != 0:
                with torch.no_grad():
                    for m in range(model_num):
                        select_id = random.sample(select_dim, feature_dim)
                        q=query_split.clone()
                        q[:,select_id]=0
                        pred = linear_clf(q)
                        if m ==0:
                            r=pred.detach().cpu().numpy()
                        else:
                            r+=pred.detach().cpu().numpy()
                # print(pred.shape)
                res_.append(r)


        res=[np.mean(r_.argmax(axis=1) == y_query.cpu().numpy()) * 100 for r_ in res_]

        acc_all1.append(res[0])
        acc_all2.append(res[1])
        acc_all3.append(res[2])
        acc_mean1 = np.mean(acc_all1)
        acc_mean2 = np.mean(acc_all2)
        acc_mean3 = np.mean(acc_all3)
        print(cot, acc_mean1, acc_mean2, acc_mean3)

        if cot==599:
            break

    acc_std1 = np.std(acc_all1)
    acc_std2 = np.std(acc_all2)
    acc_std3 = np.std(acc_all3)
    iter_num = cot+1
    print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean1, 1.96 * acc_std1 / np.sqrt(iter_num)))
    print('%d Test Acc at 200= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean2, 1.96 * acc_std2 / np.sqrt(iter_num)))
    print('%d Test Acc at 300= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean3, 1.96 * acc_std3 / np.sqrt(iter_num)))

    return 0, 0, acc_mean2, acc_std2

from sklearn.linear_model import LogisticRegression
def evaluate_shotLR(index, train_features, val_features, novel_features, few_shot_meta_data, model = None, transductive = False):
    args.preprocessing='ME'
    novel_features = preprocess(train_features, novel_features, elements_train=few_shot_meta_data["elements_train"])
    acc_list = []
    for cot,i in enumerate(few_shot_meta_data["novel_run_classes"][index]):

        z_support = novel_features[i,:n_shots]
        z_query = novel_features[i,n_shots:]

        n_support =z_support.shape[1]
        n_query=z_query.shape[1]
        n_ways=len(i)


        z_support   = z_support.view(n_support*n_ways, -1 ).cpu().numpy()
        z_query     = z_query.view(n_query*n_ways, -1 ).cpu().numpy()

        y_support = np.repeat(range( n_ways ), n_support )
        #print(y_support.shape)
        #y_support = y_support.to(args.device)

        y_query = np.repeat(range(n_ways), n_query)

        #y_query = Variable(y_query.cuda())

        classifier = LogisticRegression(max_iter=1000).fit(X=z_support, y=y_support)

        predicts = classifier.predict(z_query)
        acc = np.mean(predicts == y_query)
        acc_list.append(acc)
        if cot%100==0:
            print('%d way %d shot  ACC : %f ± %f %%'%(n_ways,n_support,float(np.mean(acc_list)),1.96 * np.std(acc_list)/np.sqrt(cot+1)*100))


    return 0, 0, np.mean(acc_list), 1.96 * np.std(acc_list)/np.sqrt(len(few_shot_meta_data["novel_run_classes"][index]))


def evaluate_shotOurs(index, train_features, val_features, novel_features, few_shot_meta_data, model=None,
                       transductive=False):
    # n_shots_n_querie=600
    # run_indices = torch.LongTensor(n_runs, few_shot_meta_data["novel_run_indices"][index].shape[1], n_shots_n_querie).to(args.device)
    # for i in range(n_runs):
    #     for j in range(few_shot_meta_data["novel_run_indices"][index].shape[1]):
    #         run_indices[i,j] = torch.randperm(n_shots_n_querie)[:n_shots_n_querie]
    args.preprocessing = 'M'
    args.postprocessing='M'
    novel_features = preprocess(train_features, novel_features, elements_train=few_shot_meta_data["elements_train"])

    acc_all1, acc_all2, acc_all3 = [], [], []

    for cot in range(n_runs // batch_few_shot_runs):
        runs = generate_runs(novel_features, few_shot_meta_data["novel_run_classes"][index],
                             few_shot_meta_data["novel_run_indices"][index], cot)
        # runs = generate_runs(novel_features, few_shot_meta_data["novel_run_classes"][index],
        #                      run_indices, cot)

        #runs=postprocess(runs)

        if transductive:
            method = 2
            dim = novel_features.shape[2]
            with torch.no_grad():
                if method==1:
                    means = torch.mean(runs[:, :, :n_shots], dim=2)
                    for i in range(500):
                        similarities = torch.norm(
                            runs[:, :, n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1,
                                                                                                          args.n_ways, dim), dim=3,
                            p=2)
                        new_allocation = (similarities == torch.min(similarities, dim=2, keepdim=True)[0]).float()
                        new_allocation = new_allocation / new_allocation.sum(dim=1, keepdim=True)
                        allocation = new_allocation
                        means = (runs[:, :, :n_shots].mean(dim=2) * n_shots + torch.einsum("rsw,rsd->rwd", allocation,
                                                                                           runs[:, :, n_shots:].reshape(
                                                                                               runs.shape[0], -1,
                                                                                               runs.shape[3])) * args.n_queries) / runs.shape[2]
                else:
                    means = torch.mean(runs[:, :, :n_shots], dim=2)
                    for i in range(30):
                        similarities = torch.norm(
                            runs[:, :, n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(
                                batch_few_shot_runs, 1, args.n_ways, dim), dim=3, p=2)
                        soft_allocations = F.softmax(-similarities.pow(2) * args.transductive_temperature_softkmeans,
                                                     dim=2)
                        means = torch.sum(runs[:, :, :n_shots], dim=2) + torch.einsum("rsw,rsd->rwd", soft_allocations,
                                                                                      runs[:, :, n_shots:].reshape(
                                                                                          runs.shape[0], -1,
                                                                                          runs.shape[3]))
                        means = means / (n_shots + soft_allocations.sum(dim=1).reshape(batch_few_shot_runs, -1, 1))

        runs = runs.squeeze(0)
        z_support = runs[:, :n_shots]

        z_query = runs[:, n_shots:]

        n_support = z_support.shape[1]
        n_query = z_query.shape[1]
        n_ways = z_query.shape[0]

        #print(z_support.shape,z_query.shape)
        z_support = z_support.contiguous().view(n_support * n_ways, -1).to(args.device)
        z_query = z_query.contiguous().view(n_query * n_ways, -1).to(args.device)
        if transductive:
            with torch.no_grad():
                z_support=torch.mean(z_support.view(n_ways,n_support,-1),dim=1)
                z_support = torch.cat((z_support, means.squeeze(0).clone()), dim=0)
                y_support=torch.cat((torch.from_numpy(np.repeat(range(n_ways), 1))
                                     , torch.from_numpy(np.repeat(range(n_ways), 1))), dim=0)
                z_support=z_support.to(args.device)
                y_support=y_support.to(args.device)
                ###
                # z_support = torch.mean(z_support.view(n_ways, n_support, -1), dim=1)
                # z_support = torch.mean(torch.cat((z_support.view(n_ways, 1, -1), means.squeeze(0).view(n_ways, 1, -1).clone()), dim=1),dim=1).to(args.device)
                # y_support=torch.from_numpy(np.repeat(range(n_ways), 1)).to(args.device)
                ###
                # z_support=means.squeeze(0).to(args.device)
                # y_support=torch.from_numpy(np.repeat(range(n_ways), 1)).to(args.device)
                #print(y_support)
        else:
            y_support=torch.from_numpy(np.repeat(range(n_ways), n_support)).to(args.device)

        n_support = z_support.shape[0]//n_ways
        # print(y_support.shape)
        # y_support = y_support.to(args.device)

        y_query = torch.from_numpy(np.repeat(range(n_ways), n_query)).to(args.device)

        # y_query = Variable(y_query.cuda())
        feat_dim = z_support.shape[1]
        # drop_rate = 0.8
        # feature_dim = int(drop_rate * feat_dim)
        model_num = 5

        select_dim = np.array(range(feat_dim)).reshape(5, -1)
        # import random

        for l in range(model_num):
            select_id = np.delete(select_dim, l).reshape(-1)
            # start=l*feature_dim
            # end=start+feature_dim
            support_split = z_support[:, select_id]
            query_split = z_query[:, select_id]

            linear_clf = distLinear(support_split.shape[1], n_ways)
            linear_clf = linear_clf.to(args.device)

            # set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
            #                                 weight_decay=0.001)
            set_optimizer = torch.optim.Adam(linear_clf.parameters(), lr=0.01,
                                             weight_decay=0.001)

            loss_function = nn.CrossEntropyLoss()
            loss_function = loss_function.to(args.device)

            batch_size = 4
            support_size = n_ways * n_support
            res_ = []
            for epoch in range(61):
                rand_id = np.random.permutation(support_size)
                for i in range(0, support_size, batch_size):
                    set_optimizer.zero_grad()
                    selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                    z_batch = support_split[selected_id]
                    y_batch = y_support[selected_id]
                    scores = linear_clf(z_batch)
                    loss = loss_function(scores, y_batch)
                    loss.backward()
                    set_optimizer.step()
                if epoch % 20 == 0 and epoch != 0:
                    pred = linear_clf(query_split)
                    # print(pred.shape)
                    res_.append(pred.detach().cpu().numpy())
            res_ = np.array(res_)
            if l == 0:
                r = res_.copy()
            else:
                r += res_

            del loss_function, linear_clf, set_optimizer

        res = [np.mean(r_.argmax(axis=1) == y_query.cpu().numpy()) * 100 for r_ in r]

        acc_all1.append(res[0])
        acc_all2.append(res[1])
        acc_all3.append(res[2])
        acc_mean1 = np.mean(acc_all1)
        acc_mean2 = np.mean(acc_all2)
        acc_mean3 = np.mean(acc_all3)
        acc_std2 = np.std(acc_all2)
        print(cot, acc_mean1, acc_mean2, acc_mean3,1.96 * acc_std2 / np.sqrt(cot + 1))

        # if cot == 799:
        #     break

    acc_std1 = np.std(acc_all1)
    acc_std2 = np.std(acc_all2)
    acc_std3 = np.std(acc_all3)
    iter_num = cot + 1
    print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean1, 1.96 * acc_std1 / np.sqrt(iter_num)))
    print('%d Test Acc at 200= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean2, 1.96 * acc_std2 / np.sqrt(iter_num)))
    print('%d Test Acc at 300= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean3, 1.96 * acc_std3 / np.sqrt(iter_num)))

    return 0, 0, acc_mean2, acc_std2


def evaluate_shotfinal(index, train_features, val_features, novel_features, few_shot_meta_data, model=None,
                  transductive=False):
    args.preprocessing = 'M'
    novel_features = preprocess(train_features, novel_features, elements_train=few_shot_meta_data["elements_train"])
    acc_all1, acc_all2, acc_all3 = [], [], []

    for cot, i in enumerate(few_shot_meta_data["novel_run_classes"][index]):

        z_support = novel_features[i, :n_shots]
        z_query = novel_features[i, n_shots:]

        n_support = z_support.shape[1]
        n_query = z_query.shape[1]
        n_ways = len(i)

        z_support = z_support.view(n_support * n_ways, -1).to(args.device)
        z_query = z_query.view(n_query * n_ways, -1).to(args.device)

        y_support = torch.from_numpy(np.repeat(range(n_ways), n_support)).to(args.device)
        # print(y_support.shape)
        # y_support = y_support.to(args.device)

        y_query = torch.from_numpy(np.repeat(range(n_ways), n_query)).to(args.device)

        # y_query = Variable(y_query.cuda())
        feat_dim = z_support.shape[1]
        # drop_rate = 0.8
        # feature_dim = int(drop_rate * feat_dim)
        model_num=5


        select_dim = np.array(range(feat_dim)).reshape(5,-1)
        #import random

        for l in range(model_num):
            select_id = np.delete(select_dim,l).reshape(-1)
            # start=l*feature_dim
            # end=start+feature_dim
            support_split = z_support[:, select_id]
            query_split = z_query[:, select_id]

            linear_clf = distLinear(support_split.shape[1], n_ways)
            linear_clf = linear_clf.to(args.device)

            # set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
            #                                 weight_decay=0.001)
            set_optimizer = torch.optim.Adam(linear_clf.parameters(), lr=0.01,
                                             weight_decay=0.001)

            loss_function = nn.CrossEntropyLoss()
            loss_function = loss_function.to(args.device)

            batch_size = 4
            support_size = n_ways * n_support
            res_ = []
            for epoch in range(61):
                rand_id = np.random.permutation(support_size)
                for i in range(0, support_size, batch_size):
                    set_optimizer.zero_grad()
                    selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                    z_batch = support_split[selected_id]
                    y_batch = y_support[selected_id]
                    scores = linear_clf(z_batch)
                    loss = loss_function(scores, y_batch)
                    loss.backward()
                    set_optimizer.step()
                if epoch % 20 == 0 and epoch != 0:
                    pred = linear_clf(query_split)
                    # print(pred.shape)
                    res_.append(pred.detach().cpu().numpy())
            res_=np.array(res_)
            if l==0:
                r=res_.copy()
            else:
                r+=res_

            del loss_function,linear_clf,set_optimizer

        res=[np.mean(r_.argmax(axis=1) == y_query.cpu().numpy()) * 100 for r_ in r]

        acc_all1.append(res[0])
        acc_all2.append(res[1])
        acc_all3.append(res[2])
        acc_mean1 = np.mean(acc_all1)
        acc_mean2 = np.mean(acc_all2)
        acc_mean3 = np.mean(acc_all3)
        print(cot, acc_mean1, acc_mean2, acc_mean3)

        if cot==599:
            break

    acc_std1 = np.std(acc_all1)
    acc_std2 = np.std(acc_all2)
    acc_std3 = np.std(acc_all3)
    iter_num = cot+1
    print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean1, 1.96 * acc_std1 / np.sqrt(iter_num)))
    print('%d Test Acc at 200= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean2, 1.96 * acc_std2 / np.sqrt(iter_num)))
    print('%d Test Acc at 300= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean3, 1.96 * acc_std3 / np.sqrt(iter_num)))

    return 0, 0, acc_mean2, acc_std2

def evaluate_shot41dropout(index, train_features, val_features, novel_features, few_shot_meta_data, model=None,
                  transductive=False):
    args.preprocessing = 'ME'
    novel_features = preprocess(train_features, novel_features, elements_train=few_shot_meta_data["elements_train"])
    acc_all1, acc_all2, acc_all3 = [], [], []

    for cot, i in enumerate(few_shot_meta_data["novel_run_classes"][index]):

        z_support = novel_features[i, :n_shots]
        z_query = novel_features[i, n_shots:]

        n_support = z_support.shape[1]
        n_query = z_query.shape[1]
        n_ways = len(i)

        z_support = z_support.view(n_support * n_ways, -1).to(args.device)
        z_query = z_query.view(n_query * n_ways, -1).to(args.device)

        y_support = torch.from_numpy(np.repeat(range(n_ways), n_support)).to(args.device)
        # print(y_support.shape)
        # y_support = y_support.to(args.device)

        y_query = torch.from_numpy(np.repeat(range(n_ways), n_query)).to(args.device)

        # y_query = Variable(y_query.cuda())
        feat_dim = z_support.shape[1]
        drop_rate = 0.8
        feature_dim = int(drop_rate * feat_dim)
        model_num=4


        select_dim = range(feat_dim)
        import random

        for l in range(model_num):
            select_id = random.sample(select_dim, feature_dim)
            # start=l*feature_dim
            # end=start+feature_dim
            support_split = z_support[:, select_id]
            query_split = z_query[:, select_id]

            linear_clf = distLinear(support_split.shape[1], n_ways)
            linear_clf = linear_clf.to(args.device)

            set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                            weight_decay=0.001)

            loss_function = nn.CrossEntropyLoss()
            loss_function = loss_function.to(args.device)

            batch_size = 4
            support_size = n_ways * n_support
            res_ = []
            for epoch in range(301):
                rand_id = np.random.permutation(support_size)
                for i in range(0, support_size, batch_size):
                    set_optimizer.zero_grad()
                    selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                    z_batch = support_split[selected_id]
                    y_batch = y_support[selected_id]
                    scores = linear_clf(z_batch)
                    loss = loss_function(scores, y_batch)
                    loss.backward()
                    set_optimizer.step()
                if epoch % 100 == 0 and epoch != 0:
                    pred = linear_clf(query_split)
                    # print(pred.shape)
                    res_.append(pred.detach().cpu().numpy())
            res_=np.array(res_)
            if l==0:
                r=res_.copy()
            else:
                r+=res_


        res=[np.mean(r_.argmax(axis=1) == y_query.cpu().numpy()) * 100 for r_ in r]

        acc_all1.append(res[0])
        acc_all2.append(res[1])
        acc_all3.append(res[2])
        acc_mean1 = np.mean(acc_all1)
        acc_mean2 = np.mean(acc_all2)
        acc_mean3 = np.mean(acc_all3)
        print(cot, acc_mean1, acc_mean2, acc_mean3)

    acc_std1 = np.std(acc_all1)
    acc_std2 = np.std(acc_all2)
    acc_std3 = np.std(acc_all3)
    iter_num = len(few_shot_meta_data["novel_run_indices"][index])
    print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean1, 1.96 * acc_std1 / np.sqrt(iter_num)))
    print('%d Test Acc at 200= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean2, 1.96 * acc_std2 / np.sqrt(iter_num)))
    print('%d Test Acc at 300= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean3, 1.96 * acc_std3 / np.sqrt(iter_num)))

    return 0, 0, acc_mean2, acc_std2

print("eval_few_shot, ", end='')
