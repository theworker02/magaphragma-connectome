"""Small reproducible 3-D boundary-model benchmark on legitimate CREMI labels.

It is deliberately a held-out benchmark, not a Megaphragma production model.
"""
import json, random, sys, time
from pathlib import Path
import h5py, numpy as np, tensorflow as tf

# This benchmark is intentionally isolated from every target-domain production decision.
root=Path(sys.argv[1]); manifest=json.loads((root/'phase5c/cremi_splits.json').read_text()); out=root/'local_research_build/phase5c-cremi'; out.mkdir(parents=True,exist_ok=True)
np.random.seed(17); random.seed(17); tf.random.set_seed(17)
# Sample only the declared CREMI split; no Megaphragma data enters this evaluator.
def sample(path, n):
  with h5py.File(path,'r') as f:
   raw=f['volumes/raw']; lab=f['volumes/labels/neuron_ids']; items=[]
   for _ in range(n):
    z=np.random.randint(1,raw.shape[0]-17); y=np.random.randint(1,raw.shape[1]-65); x=np.random.randint(1,raw.shape[2]-65)
    r=np.asarray(raw[z:z+16,y:y+64,x:x+64],np.float32)/255.; l=np.asarray(lab[z:z+16,y:y+64,x:x+64])
    b=np.zeros_like(l,dtype=bool); b[:-1]|=(l[:-1]!=l[1:]); b[:,:-1]|=(l[:,:-1]!=l[:,1:]); b[:,:,:-1]|=(l[:,:,:-1]!=l[:,:,1:])
    items.append((r[...,None],b.astype(np.float32)[...,None]))
  return np.stack([a for a,b in items]),np.stack([b for a,b in items])
train_path=root/manifest['splits']['train'][0]['path']; val_path=root/manifest['splits']['validation'][0]['path']; test_path=root/manifest['splits']['test'][0]['path']
x,y=sample(train_path,8); xv,yv=sample(val_path,4); xt,yt=sample(test_path,4)
# A deliberately small model establishes external-domain compatibility, not target fitness.
inputs=tf.keras.Input((16,64,64,1)); a=tf.keras.layers.Conv3D(8,3,padding='same',activation='relu')(inputs); a=tf.keras.layers.Conv3D(8,3,padding='same',activation='relu')(a); outputs=tf.keras.layers.Conv3D(1,1,activation='sigmoid')(a); model=tf.keras.Model(inputs,outputs); model.compile(optimizer='adam',loss='binary_crossentropy')
started=time.time(); history=model.fit(x,y,validation_data=(xv,yv),epochs=1,batch_size=1,verbose=0); pred=model.predict(xt,verbose=0)>0.5
tp=int(np.logical_and(pred,yt>0.5).sum()); fp=int(np.logical_and(pred,yt<=0.5).sum()); fn=int(np.logical_and(~pred,yt>0.5).sum()); dice=(2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.
model.save_weights(out/'weights.weights.h5')
# The receipt encodes the rejection so a numerical benchmark cannot be mistaken for selection.
result={'id':'MV-MODEL-CREMI-BOUNDARY-0001','kind':'3D_BOUNDARY_MODEL','status':'TRAINED_EXTERNAL_BENCHMARK_ONLY','train':'CREMI-A','validation':'CREMI-B','test':'CREMI-C','epochs':1,'patch_zyx':[16,64,64],'seed':17,'runtime_seconds':time.time()-started,'held_out_test':{'boundary_dice':dice,'tp':tp,'fp':fp,'fn':fn},'target_compatibility':'REJECTED_RESOLUTION_AND_MODALITY_MISMATCH','production_selection':'NOT_SELECTED','reason':'CREMI is 40x4x4 nm Drosophila ssTEM; target is 8x8x8 nm Megaphragma FIB-SEM. No target-compatible held-out labels exist.','weights':'weights.weights.h5'}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
