"""Scientific invariants, leakage boundaries, retrieval metrics and figure smoke tests."""
from __future__ import annotations

import json
import numpy as np
import pytest

from curse_of_dim.advanced import (
    effective_rank, knn_overlap, swiss_roll, embedding_transformations,
    retrieval_metrics, synthetic_embeddings, experiment_08_nonlinear_manifold,
    experiment_09_anisotropic_embeddings, experiment_10_effective_dimension,
)
from curse_of_dim.audit import audit_npz


def test_rank_orthogonal_low_rank_and_zero():
    rng=np.random.default_rng(4)
    z=rng.normal(size=(500,3))
    basis,_=np.linalg.qr(rng.normal(size=(100,3)))
    ranks=effective_rank(z@basis.T)
    assert 2.6 < ranks['entropy_rank'] < 3.1
    assert ranks['participation_ratio'] <= ranks['entropy_rank']+1e-8
    assert effective_rank(np.ones((10,5)))['entropy_rank']==0.0


def test_knn_overlap_isometry():
    x, oracle, _=swiss_roll(130, 32, seed=1)
    assert knn_overlap(x[:100], x[100:], x[:100], x[100:], 5)==1.0
    assert 0<=knn_overlap(oracle[:100],oracle[100:],x[:100],x[100:],5)<=1


def test_centering_and_top_pc_fit_docs_not_queries():
    rng=np.random.default_rng(11)
    doc=rng.normal(size=(80,8))+5
    q=rng.normal(size=(13,8))+5
    transformed=embedding_transformations(doc,q)
    q_ext=np.vstack([q,np.full((2,8),10000)])
    transformed_ext=embedding_transformations(doc,q_ext)
    for key in transformed:
        np.testing.assert_allclose(transformed[key][0],transformed_ext[key][0],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(transformed[key][1],transformed_ext[key][1][:len(q)],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(transformed['centered'][0].mean(axis=0),0,atol=1e-12)


def test_retrieval_label_oracle_and_invalid_k():
    docs=np.eye(8)
    q=docs[:3].copy()
    labels=np.arange(8)
    report=retrieval_metrics(docs,q,labels,labels[:3],k=1)
    assert report['precision_at_k']==1
    assert report['top1_accuracy']==1
    assert report['hit_at_k']==1
    with pytest.raises(ValueError):
        retrieval_metrics(docs,q,k=9)


def test_top_pc_removal_counterexample():
    d,q,dl,ql=synthetic_embeddings(scenario='signal')
    trans=embedding_transformations(d,q)
    raw=retrieval_metrics(*trans['raw'],dl,ql,k=5)['top1_accuracy']
    removed=retrieval_metrics(*trans['remove_top_pc'],dl,ql,k=5)['top1_accuracy']
    assert raw > .9
    assert removed < raw-.2


def test_audit_npz_does_not_need_external_models(tmp_path):
    d,q,dl,ql=synthetic_embeddings(n_docs=80,n_queries=25,ambient=32)
    data=tmp_path/'vectors.npz'
    np.savez_compressed(data,doc_embeddings=d,query_embeddings=q,doc_labels=dl,query_labels=ql)
    result=audit_npz(data,tmp_path/'audit',k=3)
    assert result['samples']['dimension']==32
    assert (tmp_path/'audit'/'audit.png').stat().st_size>4000
    assert json.loads((tmp_path/'audit'/'audit.json').read_text())['k']==3


def test_advanced_experiments_create_figures(tmp_path):
    a=experiment_08_nonlinear_manifold(tmp_path)
    b=experiment_09_anisotropic_embeddings(tmp_path)
    c=experiment_10_effective_dimension(tmp_path)
    assert a['intrinsic_dimension']==2
    assert b['nuisance']['raw']['precision_at_k']<=1
    assert c['true_latent_dimension']==6
    for name in ('08_nonlinear_manifold.png','09_anisotropy_retrieval.png','10_effective_dimension.png'):
        assert (tmp_path/name).stat().st_size>5000
