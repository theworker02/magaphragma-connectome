# Google FFN bootstrap

The official upstream source was acquired in isolated `third_party/ffn` at commit `b8df2d96d7c8da057fd2702ced19732a08ab6ae4`. It uses TensorFlow compatibility-v1 APIs and also currently imports JAX and Connectomics through the inference stack.

An isolated Python 3.13.14 environment imports TensorFlow 2.21.0 and `ffn.inference.executor` successfully on CPU. This is **ENVIRONMENT_IMPORT_PASS**, not inference success.

No trained checkpoint was acquired. The checkpoint configured by upstream’s example is `models/fib25/model.ckpt-27465036`; the README describes authenticated Google Cloud Storage retrieval. A direct public probe returned HTTP 404 on 2026-09-16. Therefore the state is **CHECKPOINT_UNAVAILABLE**, reference inference is not run, and target compatibility/inference are not assessable.

The current DVID-native watershed output is not FFN and must never be represented as FFN output. The next executable FFN action is to obtain an official, checksum-verifiable checkpoint plus its model and training-data provenance.
