import torch

from worker import *


def test_overlap(w: Worker):
    # test overlap
    b = 1
    d_model = 4096
    intermediate_dimension = 11008
    dtype = torch.float32
    split_size = d_model // world_size
    xi = torch.randn(split_size, dtype=dtype)
    print(xi)

    # test ring_reduce_scatter_overlap()
    Wi = torch.randn(split_size, d_model, dtype=dtype)
    print(Wi)
    yi = w.ring_reduce_scatter_comp_overlap(xi, Wi)
    print(yi)
    # verify result
    xis = [torch.empty_like(xi, dtype=dtype)] * world_size
    dist.all_gather(xis, xi)
    for x in xis:
        print(torch.equal(x, xi))
    x = torch.concat(xis, dim=0)
    Wis = [torch.empty_like(Wi, dtype=dtype)] * world_size
    dist.all_gather(Wis, Wi)  # bug
    Wis[rank] = Wi
    Wis = torch.concat(Wis, dim=0)
    print(Wis)
    Wi = torch.split(Wis, split_size, dim=-1)[rank]
    yi_correct = x @ Wi
    print(torch.allclose(yi_correct, yi, atol=1e-3))  # >> True
    print(torch.equal(yi_correct, yi))  # >> False
    print(yi_correct)

    # test ring_reduce_scatter_overlap()
    Wi = torch.randn(d_model, intermediate_dimension // world_size, dtype=dtype)
    print(Wi)
    yi = w.ring_gather_reduce_comp_overlap(xi, Wi)
    print(yi)
    # verify result
    xis = [torch.empty_like(xi, dtype=dtype)] * world_size
    dist.all_gather(xis, xi)  # bug
    xis[rank] = xi
    x = torch.concat(xis, dim=0)
    yi_correct = x @ Wi
    print(torch.allclose(yi_correct, yi, atol=1e-3))  # >> True
    print(torch.equal(yi_correct, yi))  # >> False
    print(yi_correct)


def test_ring_comm(w: Worker):
    xi = torch.randn(4)
    # real all-reduce
    x_all_reduce_correct = xi.clone()
    dist.all_reduce(x_all_reduce_correct, op=dist.ReduceOp.SUM)

    # my all-reduce
    x_all_reduce = w.all_reduce(xi)

    print(torch.allclose(x_all_reduce_correct, x_all_reduce))


if __name__ == '__main__':
    worker = Worker((MAIN_WORKER_IP, port_tcp))
    # test_overlap(worker)
    test_ring_comm(worker)
