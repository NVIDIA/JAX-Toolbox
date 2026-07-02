---
title: FAQ
subtitle: Frequently asked questions
slug: faq
---

## `bus error` when running JAX in a docker container

**Solution:**

```bash
docker run -it --shm-size=1g ...
```

**Explanation:** The `bus error` might occur due to the size limitation of
`/dev/shm`. You can address this by increasing the shared memory size using the
`--shm-size` option when launching your container.

## enroot/pyxis reports error code 404 when importing multi-arch images

**Problem description:**

```
slurmstepd: error: pyxis:     [INFO] Authentication succeeded
slurmstepd: error: pyxis:     [INFO] Fetching image manifest list
slurmstepd: error: pyxis:     [INFO] Fetching image manifest
slurmstepd: error: pyxis:     [ERROR] URL https://ghcr.io/v2/nvidia/jax/manifests/<TAG> returned error code: 404 Not Found
```

**Solution:** Upgrade [enroot](https://github.com/NVIDIA/enroot) or
[apply a single-file patch](https://github.com/NVIDIA/enroot/releases/tag/v3.4.0)
as mentioned in the enroot v3.4.0 release note.

**Explanation:** Docker has traditionally used Docker Schema V2.2 for multi-arch
manifest lists but has switched to using the Open Container Initiative (OCI)
format since 20.10. Enroot added support for OCI format in version 3.4.0.
