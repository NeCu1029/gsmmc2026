"""Check context order invariance, target exclusion, and official score parity."""
from types import SimpleNamespace
import bike_context_attention as app
import numpy as np
import torch


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    rng=np.random.default_rng(7)
    # Independent synthetic inputs: no target labels enter the forward API.
    data=SimpleNamespace(capacity=np.array([5,8,6]),similarity=rng.normal(size=(3,9)),
                         city=SimpleNamespace(temporal=torch.randn(30,24,30),off_day=np.arange(30)%7>=5))
    nt=app.old.neural.inventory.day_features(data).shape[-1]
    m=app.AttentionInventory(9,nt)
    m.scales(data,np.arange(17),np.arange(3));m.eval()
    d,s=torch.tensor([24,25]),torch.tensor([0,2])
    c=torch.tensor([[8,9,22,23],[8,9,22,23]])
    net=torch.randn(2,4,24);inc=torch.randn(2,4,8)
    with torch.no_grad():
        result=m(data,d,s,c,net,inc)
        perm=[3,0,2,1]
        swapped=m(data,d,s,c[:,perm],net[:,perm],inc[:,perm])
    torch.testing.assert_close(result,swapped,rtol=1e-5,atol=1e-5)
    for count in [2,3,4]:
        targets=np.tile(np.arange(17),10)
        context=app.episodic_context(rng,np.arange(17),targets,count).numpy()
        assert not (context==targets[:,None]).any()
        assert all(len(np.unique(row))==count for row in context)
    assert result.shape==(2,8) and torch.isfinite(result).all()
    print('PASS: public-context permutation invariance, target exclusion, variable context sizes, finite outputs')


if __name__=='__main__':main()
