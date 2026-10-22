#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Curse of dimensionality: seven reproducible experiments.
Dependencies: numpy, scipy, scikit-learn, matplotlib, threadpoolctl.
Run: python curse_dimensionality_lab.py --out figures
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error
from threadpoolctl import threadpool_limits
from . import advanced as _advanced

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.16,
    "font.size": 10, "axes.titlesize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
OUTPUT_DIR = Path("figures")


def save(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight", facecolor="white")
    print("saved:", OUTPUT_DIR / filename)
    plt.close(fig)


def experiment_01_geometry():
    """Fixed grid resolution, central volume, and nearest-neighbor L_inf radius."""
    rng = np.random.default_rng(10)
    d = np.arange(1, 101)
    n = 1000
    # On a periodic unit torus, L_inf ball radius r has volume (2r)^d.
    # P(R_min <= r) = 1-(1-(2r)^d)^n; median obtained by setting P=.5.
    p_cell = -np.expm1(np.log(0.5) / n)
    median_r = 0.5 * np.exp(np.log(p_cell) / d)
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.1), constrained_layout=True)
    xs = rng.uniform(size=(130, 2))
    ax[0,0].scatter(xs[:,0], xs[:,1], s=15, alpha=.72)
    for z in np.linspace(0,1,21):
        ax[0,0].axvline(z, lw=.45, alpha=.20)
        ax[0,0].axhline(z, lw=.45, alpha=.20)
    ax[0,0].set(xlim=(0,1), ylim=(0,1), aspect='equal',
                 title='2D: 130 samples in 400 grid cells', xlabel='x1', ylabel='x2')
    ax[0,1].plot(d, d*np.log10(10), lw=2)
    ax[0,1].set(title='Fixed resolution: 10 cells per coordinate',
                xlabel='Ambient dimension d', ylabel='log10(number of grid cells)')
    ax[1,0].semilogy(d, 0.8**d, lw=2)
    ax[1,0].set(title='Central hypercube volume fraction: 0.8^d',
                xlabel='Dimension d', ylabel='Volume fraction (log scale)')
    ax[1,1].plot(d, median_r, lw=2)
    ax[1,1].axhline(.5, ls='--', alpha=.5)
    ax[1,1].set(xlim=(1,100), ylim=(0,.52),
                title='Median nearest-neighbor L_inf radius (n=1000)',
                xlabel='Dimension d', ylabel='Radius on periodic unit cube')
    fig.suptitle('Experiment 1 | Geometry and coverage', fontsize=15, weight='bold')
    save(fig, '01_geometry.png')
    return {'r_med_d2':float(median_r[1]),'r_med_d100':float(median_r[99])}


def experiment_02_distances():
    """Concentration holds for independent uniform features; not all high-D data."""
    rng = np.random.default_rng(20)
    ds = [2, 5, 10, 20, 50, 100, 256, 512]
    cvs, ratios, hist = [], [], {}
    for d in ds:
        a = rng.random((3000, d), dtype=np.float32)
        b = rng.random((3000, d), dtype=np.float32)
        distances = np.sqrt(np.sum((a-b)**2, axis=1, dtype=np.float64))
        cvs.append(float(distances.std()/distances.mean()))
        if d in [2, 10, 100, 512]:
            hist[d] = distances / np.sqrt(d/6)
        base = rng.random((350, d), dtype=np.float32)
        queries = rng.random((50, d), dtype=np.float32)
        # Sum of coordinate-wise squared differences: shape [query, database].
        dmat = np.sqrt(np.sum((queries[:,None,:]-base[None,:,:])**2,
                               axis=2, dtype=np.float64))
        ratios.append(float(np.mean(dmat.min(axis=1)/dmat.max(axis=1))))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for d in [2, 10, 100, 512]:
        ax[0].hist(hist[d], bins=np.linspace(0,2.1,65), density=True,
                   histtype='step', lw=2, label=f'd={d}')
    ax[0].set(title='Pairwise distance / sqrt(d/6)', xlabel='Normalized Euclidean distance',
              ylabel='Density', xlim=(0,2.1))
    ax[0].legend(frameon=False)
    ax[1].loglog(ds, cvs, 'o-', label='Simulation CV(D)')
    theory = np.sqrt(7/(20*np.array(ds,dtype=float)))
    ax[1].loglog(ds, theory, '--', label='Delta-method ~sqrt(7/(20d))')
    ax[1].set(title='Distance relative fluctuations shrink',
              xlabel='Dimension d (log)', ylabel='std(D) / mean(D) (log)')
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].semilogx(ds, ratios, 'o-')
    ax[2].set(xlim=(1,600), ylim=(0,1.03),
              title='Nearest / farthest distance (fixed n=350)',
              xlabel='Dimension d (log)', ylabel='mean(min D / max D)')
    fig.suptitle('Experiment 2 | Distance concentration (i.i.d. Uniform[0,1])',
                 fontsize=15, weight='bold')
    save(fig, '02_distances.png')
    return {'cv_d2':cvs[0],'cv_d512':cvs[-1], 'near_far_d2':ratios[0], 'near_far_d512':ratios[-1]}


def generate_classification_data(seed=30, ntrain=900, ntest=400, dmax=256):
    rng = np.random.default_rng(seed)
    tr = rng.uniform(-1, 1, size=(ntrain, dmax))
    te = rng.uniform(-1, 1, size=(ntest, dmax))
    clean_tr = ((tr[:,0]+tr[:,1])>0).astype(int)
    clean_te = ((te[:,0]+te[:,1])>0).astype(int)
    flips_tr = rng.random(ntrain)<.05
    flips_te = rng.random(ntest)<.05
    return tr, te, np.logical_xor(clean_tr,flips_tr).astype(int), np.logical_xor(clean_te,flips_te).astype(int)


def experiment_03_knn_noise():
    """Two useful features plus d-2 irrelevant features of equal marginal variance."""
    dims = [2,4,8,16,32,64,128,256]
    names = ['KNN all features','KNN oracle 2 features',
             'SelectKBest(2) + KNN','LogisticRegression']
    scores = np.zeros((2,len(dims),len(names)))
    for j, seed in enumerate([30,31]):
        X, Xt, y, yt = generate_classification_data(seed)
        oracle = KNeighborsClassifier(n_neighbors=11).fit(X[:,:2],y)
        oracle_accuracy = accuracy_score(yt,oracle.predict(Xt[:,:2]))
        for i,d in enumerate(dims):
            x, xt = X[:,:d], Xt[:,:d]
            models = [
                KNeighborsClassifier(n_neighbors=11, algorithm='brute'),
                None,
                make_pipeline(SelectKBest(f_classif, k=2), KNeighborsClassifier(n_neighbors=11)),
                LogisticRegression(C=1.0, max_iter=400),
            ]
            for m,model in enumerate(models):
                if m==1:
                    scores[j,i,m] = oracle_accuracy
                else:
                    model.fit(x,y)
                    scores[j,i,m] = accuracy_score(yt, model.predict(xt))
    avg = scores.mean(axis=0)
    fig,ax=plt.subplots(1,2,figsize=(13.8,4.9),constrained_layout=True)
    for j,name in enumerate(names):
        ax[0].semilogx(dims, avg[:,j], 'o-', label=name)
    ax[0].axhline(.5, ls=':', color='gray', label='Chance level')
    ax[0].set(title='Test accuracy (mean over 2 seeds)', xlabel='Ambient dimension d (log)',
              ylabel='Accuracy', ylim=(.45,1.02))
    ax[0].legend(fontsize=8,frameon=False)
    ax[1].semilogx(dims,avg[:,1]-avg[:,0],'o-',label='Oracle KNN minus raw KNN')
    ax[1].axhline(0,ls='--',color='gray')
    ax[1].set(title='Penalty for irrelevant features', xlabel='Ambient dimension d (log)',
              ylabel='Accuracy gap')
    fig.suptitle('Experiment 3 | A controlled KNN failure: add only irrelevant dimensions',
                 fontsize=14,weight='bold')
    save(fig,'03_knn_accuracy.png')

    # Conditional slices are valid visualizations, but are NOT marginal decision regions.
    X, Xt, y, yt = generate_classification_data(30)
    knn2 = KNeighborsClassifier(n_neighbors=11).fit(X[:,:2],y)
    knn64 = KNeighborsClassifier(n_neighbors=11,algorithm='brute').fit(X[:,:64],y)
    log64 = LogisticRegression(max_iter=400).fit(X[:,:64],y)
    axis = np.linspace(-1,1,61)
    x1,x2 = np.meshgrid(axis,axis)
    coords = np.column_stack((x1.ravel(),x2.ravel()))
    fixed = np.tile(Xt[0,:64],(len(coords),1))
    fixed[:,:2] = coords
    proba = [((coords[:,0]+coords[:,1])>0).astype(float),
             knn2.predict_proba(coords)[:,1],
             knn64.predict_proba(fixed)[:,1],
             log64.predict_proba(fixed)[:,1]]
    fig,axs=plt.subplots(1,4,figsize=(15,3.75),sharex=True,sharey=True,constrained_layout=True)
    for a,p,title in zip(axs,proba,['True boundary','KNN: 2D','KNN: 64D','Logistic: 64D']):
        a.imshow(p.reshape(len(axis),len(axis)),extent=(-1,1,-1,1),
                 origin='lower',vmin=0,vmax=1,cmap='coolwarm',interpolation='nearest')
        a.plot(axis,-axis,ls='--',color='black',lw=.9,alpha=.7)
        a.set(title=title,xlabel='signal x1',ylabel='signal x2',aspect='equal')
    fig.suptitle('Conditional 2D slices (64D noise coordinates fixed to one real test sample)',
                 fontsize=13,weight='bold')
    save(fig,'03_knn_slices.png')
    return {'d2_knn':float(avg[0,0]),'d256_knn':float(avg[-1,0]),
            'd256_oracle':float(avg[-1,1]),'d256_selection':float(avg[-1,2]),
            'd256_logistic':float(avg[-1,3])}


def synthetic_smooth_target(X):
    d=X.shape[1]
    return np.sqrt(2/d)*np.sin(2*np.pi*X).sum(axis=1)


def experiment_04_samples():
    """Noiseless 1-nearest-neighbor regression; finite sample MSE vs n and d."""
    rng=np.random.default_rng(40)
    dimensions=[1,2,4,8,16]
    ns=[50,100,300,900,2700]
    all_mse=[]
    for d in dimensions:
        X=rng.uniform(size=(max(ns),d))
        Xt=rng.uniform(size=(360,d))
        y=synthetic_smooth_target(X)
        yt=synthetic_smooth_target(Xt)
        values=[]
        for n in ns:
            model=KNeighborsRegressor(n_neighbors=1,algorithm='auto').fit(X[:n],y[:n])
            values.append(mean_squared_error(yt,model.predict(Xt)))
        all_mse.append(values)
    all_mse=np.asarray(all_mse)
    fig,ax=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for i,d in enumerate(dimensions):
        ax[0].loglog(ns,all_mse[i], 'o-', label=f'd={d}')
    ax[0].set(title='1-NN test MSE, noiseless smooth target',
              xlabel='Training samples n (log)',ylabel='Test MSE (log)')
    ax[0].legend(frameon=False)
    img=ax[1].imshow(np.log10(all_mse+1e-12),aspect='auto',cmap='viridis')
    ax[1].set(title='Log10(test MSE) heatmap',xticks=range(len(ns)),yticks=range(len(dimensions)),
              xticklabels=ns,yticklabels=dimensions,xlabel='Samples n',ylabel='Dimension d')
    fig.colorbar(img,ax=ax[1],label='log10 MSE',shrink=.8)
    fig.suptitle('Experiment 4 | More data is not equally useful in every dimension',
                 fontsize=14,weight='bold')
    save(fig,'04_sample_complexity.png')
    return {'mse_d1_n2700':float(all_mse[0,-1]),
            'mse_d16_n2700':float(all_mse[-1,-1])}


def experiment_05_intrinsic():
    """Orthonormal embedding 2 -> 128 preserves geometry; ambient noise destroys it."""
    rng=np.random.default_rng(50)
    ntr,nte,D,k=1000,400,128,2
    z=rng.normal(size=(ntr,k))
    zt=rng.normal(size=(nte,k))
    Q,_=np.linalg.qr(rng.normal(size=(D,k)))
    # (2 -> 128) is an isometry in the noise-free case: ||z-z'||=||(z-z')Q.T||
    xtemplate=z@Q.T
    ttemplate=zt@Q.T
    noise_train=rng.normal(size=(ntr,D))
    noise_test=rng.normal(size=(nte,D))
    def f(zz):
        return np.sin(1.5*zz[:,0]) + 0.5*zz[:,1]
    y,yt=f(z),f(zt)
    sigmas=[0,.05,.1,.2,.35,.5]
    raw,pca,oracle=[],[],[]
    spectra={}
    for sigma in sigmas:
        X=xtemplate+sigma*noise_train
        Xt=ttemplate+sigma*noise_test
        rawmodel=KNeighborsRegressor(n_neighbors=13,weights='distance').fit(X,y)
        raw.append(mean_squared_error(yt,rawmodel.predict(Xt)))
        pcamodel=PCA(n_components=2,svd_solver='randomized',random_state=0)
        Xlow=pcamodel.fit_transform(X)
        Xtlow=pcamodel.transform(Xt)
        lowmodel=KNeighborsRegressor(n_neighbors=13,weights='distance').fit(Xlow,y)
        pca.append(mean_squared_error(yt,lowmodel.predict(Xtlow)))
        oraclemodel=KNeighborsRegressor(n_neighbors=13,weights='distance').fit(z,y)
        oracle.append(mean_squared_error(yt,oraclemodel.predict(zt)))
        if sigma in [0,.2,.5]:
            spectra[sigma]=np.linalg.eigvalsh(np.cov(X,rowvar=False))[::-1]
    fig,ax=plt.subplots(1,2,figsize=(12.5,4.7),constrained_layout=True)
    ax[0].plot(sigmas,raw,'o-',label='KNN raw 128D')
    ax[0].plot(sigmas,pca,'o-',label='PCA to 2D + KNN')
    ax[0].plot(sigmas,oracle,'--',label='Oracle latent 2D + KNN')
    ax[0].set(title='Prediction: ambient noise vs recovered structure',
              xlabel='Noise std per ambient coordinate',ylabel='Test MSE')
    ax[0].legend(frameon=False,fontsize=8)
    for sigma,eig in spectra.items():
        ax[1].semilogy(np.arange(1,D+1),np.maximum(eig,1e-9), label=f'noise={sigma}')
    ax[1].set(title='Covariance eigenvalue spectrum',xlabel='Principal-component index',
              ylabel='Eigenvalue (log)')
    ax[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Experiment 5 | Ambient dimension 128, intrinsic dimension 2',
                 fontsize=14,weight='bold')
    save(fig,'05_intrinsic_dimension.png')
    return {'raw_noise0':float(raw[0]),'pca_noise0':float(pca[0]),
            'oracle_noise0':float(oracle[0]),'raw_noise05':float(raw[-1]),
            'pca_noise05':float(pca[-1]),'oracle_noise05':float(oracle[-1])}


def experiment_06_double_descent():
    """High-dimensional linear regression: curse is not universally exponential."""
    ntrain,ntest,dmax=110,500,300
    ds=[5,10,20,40,60,80,95,105,108,110,112,115,125,140,170,220,300]
    w=np.array([1,.8,-.7,.5,-.5])
    w=w/np.linalg.norm(w)
    out=np.zeros((3,len(ds),3))  # repeat x dimension x (train OLS/test OLS/test Ridge)
    sigma=.5
    for j,seed in enumerate([60,61,62]):
        rng=np.random.default_rng(seed)
        X=rng.normal(size=(ntrain,dmax))
        Xt=rng.normal(size=(ntest,dmax))
        yt_clean=Xt[:,:5]@w
        y=X[:,:5]@w+sigma*rng.normal(size=ntrain)
        yt=yt_clean+sigma*rng.normal(size=ntest)
        for i,d in enumerate(ds):
            x,xt=X[:,:d],Xt[:,:d]
            coef,*_=lstsq(x,y,lapack_driver='gelsd')
            pred=x@coef
            testpred=xt@coef
            ridge=Ridge(alpha=15).fit(x,y)
            out[j,i]=[mean_squared_error(y,pred),
                      mean_squared_error(yt,testpred),
                      mean_squared_error(yt,ridge.predict(xt))]
    med=np.median(out,axis=0)
    fig,ax=plt.subplots(figsize=(9.8,5.0),constrained_layout=True)
    ax.semilogy(np.array(ds)/ntrain,med[:,0]+1e-8,'o-',label='Min-norm train MSE')
    ax.semilogy(np.array(ds)/ntrain,med[:,1],'o-',label='Min-norm test MSE')
    ax.semilogy(np.array(ds)/ntrain,med[:,2],'o-',label='Ridge test MSE (alpha=15)')
    ax.axvline(1,ls='--',label='d/n=1 interpolation threshold',color='gray')
    ax.axhline(sigma**2,ls=':',label='Noise variance = 0.25',color='gray')
    ax.set(title='Linear regression: bias / variance / interpolation',
           xlabel='Number of features / number of training samples (d/n)',
           ylabel='Mean squared error (log)',ylim=(1e-6,100))
    ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Experiment 6 | Double descent and regularization (median over 3 seeds)',
                 fontsize=14,weight='bold')
    save(fig,'06_double_descent.png')
    return {'peak_mse_near_dn1':float(med[ds.index(110),1]),
            'ridge_at_dn1':float(med[ds.index(110),2])}


def experiment_07_conditioning():
    """A 2D ill-conditioned problem shows that optimization pathology != high dimension."""
    rng=np.random.default_rng(70)
    n=1200
    X=rng.normal(size=(n,2))*np.array([1,.03])
    wstar=np.array([1.,20.])
    y=X@wstar
    H=X.T@X/n
    lam=np.linalg.eigvalsh(H)
    cond=float(lam.max()/lam.min())
    sd=X.std(axis=0)
    Xwhite=X/sd

    def fit_path(x,nsteps=1600):
        eta=.9/np.linalg.eigvalsh(x.T@x/n).max()
        w=np.zeros(2)
        errors=[]
        paths=[]
        for _ in range(nsteps):
            r=x@w-y
            errors.append(np.mean(r*r)/2)
            paths.append(w.copy())
            w-=eta*(x.T@r/n)
        return np.asarray(errors),np.asarray(paths)

    e_raw,w_raw=fit_path(X)
    e_white,v_white=fit_path(Xwhite)
    w_white=v_white/sd
    fig,ax=plt.subplots(1,3,figsize=(15,4.35),constrained_layout=True)
    ax[0].semilogy(np.maximum(e_raw,1e-15),label=f'Raw coordinates; kappa~{cond:.0f}')
    ax[0].semilogy(np.maximum(e_white,1e-15),label='Whitened (diagonal scaling)')
    ax[0].set(title='Gradient descent: training objective',xlabel='GD step',ylabel='Half MSE (log)')
    ax[0].legend(frameon=False,fontsize=8)
    ax[1].plot(w_raw[:,1],label='Raw: estimated w2')
    ax[1].plot(w_white[:,1],label='Rescaled: estimated w2')
    ax[1].axhline(wstar[1],ls='--',color='gray',label='True w2')
    ax[1].set(title='Slow direction coefficient',xlabel='GD step',ylabel='w2')
    ax[1].legend(frameon=False,fontsize=8)
    w1=np.linspace(-.1,1.6,115)
    w2=np.linspace(-1,24,115)
    a,b=np.meshgrid(w1,w2)
    delta=np.stack([a-wstar[0],b-wstar[1]],axis=-1)
    objective=.5*np.einsum('...i,ij,...j->...',delta,H,delta)
    ax[2].contour(a,b,objective,levels=[.005,.02,.05,.1,.2,.4,.8,1.6,3.2],alpha=.65)
    idx=np.unique(np.round(np.geomspace(1,len(w_raw),80)).astype(int)-1)
    ax[2].plot(w_raw[idx,0],w_raw[idx,1],lw=1.6,label='Raw path')
    ax[2].plot(w_white[idx,0],w_white[idx,1],lw=1.6,label='Whitened path')
    ax[2].scatter(*wstar,s=55,marker='x',color='black',label='Optimum')
    ax[2].set(title='Loss contours and GD trajectories',xlabel='w1',ylabel='w2',
              xlim=(-.1,1.6),ylim=(-1,24))
    ax[2].legend(frameon=False,fontsize=8)
    fig.suptitle('Experiment 7 | Only TWO dimensions, yet severe ill-conditioning',
                 fontsize=14,weight='bold')
    save(fig,'07_ill_conditioning.png')
    return {'condition_number':cond,'raw_loss_end':float(e_raw[-1]),
            'whitened_loss_end':float(e_white[-1])}


def run_all(out="figures"):
    """Execute the default experiments (excluding optional torch autoencoder)."""
    return run_selected(tuple(n for n in EXPERIMENTS if n != "autoencoder_manifold"), out=out)



EXPERIMENTS = {
    "geometry": experiment_01_geometry,
    "distances": experiment_02_distances,
    "knn_noise": experiment_03_knn_noise,
    "sample_complexity": experiment_04_samples,
    "intrinsic_dimension": experiment_05_intrinsic,
    "double_descent": experiment_06_double_descent,
    "conditioning": experiment_07_conditioning,
    "nonlinear_manifold": lambda: _advanced.experiment_08_nonlinear_manifold(OUTPUT_DIR),
    "embedding_anisotropy": lambda: _advanced.experiment_09_anisotropic_embeddings(OUTPUT_DIR),
    "effective_dimension": lambda: _advanced.experiment_10_effective_dimension(OUTPUT_DIR),
    "autoencoder_manifold": lambda: _advanced.experiment_11_autoencoder(OUTPUT_DIR),
}



def run_selected(names, out="figures"):
    """Run selected experiments with deterministic RNG seeds and save figures.

    The output directory is process-global; this runner is for sequential CLI use.
    """
    global OUTPUT_DIR
    invalid = set(names) - EXPERIMENTS.keys()
    if invalid:
        raise ValueError(f"Unknown experiment(s): {sorted(invalid)}")
    OUTPUT_DIR = Path(out)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    with threadpool_limits(limits=2):
        for name in names:
            print(f"Running {name}", flush=True)
            summary[name] = EXPERIMENTS[name]()
    return summary
