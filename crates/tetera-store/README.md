# tetera-store

Tetera's dataset store, L2 (plan §6.2 and §6.9, [ADR-0021](../../docs/adr/0021-datasets-on-disk.md)): projects of datasets under the data root, and their index.

- **`Store::open(root)`** keeps projects in `root/projects/` and its index in `root/tetera.db`, beside the jobs' history. A dataset left half written by a crash is removed when the store opens.
- **Projects** have names that make folders: 1 to 64 letters, digits, spaces, dots, dashes or underscores.
- **`create_dataset`** gives a `DatasetWriter`. It writes frames one by one, and `finish` lists the dataset. Dropped before `finish`, the writer leaves nothing: no folder, no row.
- **A dataset's folder,** `projects/<project>/<id>/`:
  - `image.ome.zarr/`: OME-Zarr v3, the frames as one array `0` with axes t, c, y and x, one chunk a frame, compressed with zstd, and OME-NGFF 0.5's metadata on the group. Raw Bayer is one channel, its pattern in `metadata.json`;
  - `metadata.json`: what it holds, the instrument as it was, and each frame's metadata, its number, times, settings and stage position;
  - `thumbnail.jpg`: each 2 × 2 Bayer cell as a pixel, at most 256 across.
- **Reading back:** `read` gives a `DatasetReader`; `datasets`, `dataset`, `files`, `thumbnail`, `delete`.

## How to try it

```bash
cargo run --release -p tetera-store --example datasets
```

It writes 20 full-size raw frames into a new project, reads them back, prints the dataset's tree, and drops a second dataset half way to show it leaves nothing behind. `-- --smoke` writes small frames.
