# ADR-0021: Datasets on disk

- Status: Accepted
- Date: 2026-11-28
- Phase: P1.15b

## Context

- **Captures and scans make datasets** that people find, read, export and delete later, from Tetera and from other tools (Q36, Q37: OME-Zarr, in projects).
- **OME-Zarr** is the open format for bioimages that tools such as napari and Fiji read: OME-NGFF's metadata over a Zarr store. OME-NGFF 0.5 is on Zarr v3.
- **The camera's frames are raw Bayer,** 8 bits or 12 in 16, up to 5472 × 3648.
- **A dataset is written over seconds to hours.** A crash or a cancel can stop it half way, and no one must find it listed then.
- **The plan keeps one SQLite file,** `tetera.db` in the data root, for the datasets' index, the jobs' history and the calibration sets (plan §6.9, ADR-0020).

## Decision

1. **Projects are folders** in the data root's `projects/`, named for people; a dataset is a folder in its project, named by its id, a UUID v7.
2. **A dataset's frames are OME-Zarr v3,** through `zarrs`, in `image.ome.zarr/`:
   - one array, `0`, of axes t, c, y and x;
   - raw Bayer as one channel, its pattern in the metadata;
   - one chunk for each frame, compressed with zstd at level 3;
   - the group's attributes carry OME-NGFF 0.5's `multiscales`, with the pixel size at the sample, µm, and the time between frames, s, as its scale.
3. **Beside it, `metadata.json`:** what the dataset holds, the instrument as it was (camera settings, objective, calibration), and each frame's metadata. **And `thumbnail.jpg`:** each 2 × 2 Bayer cell as one pixel, at most 256 across, brightened by a gamma of 2.2.
4. **The index is SQLite,** in `tetera.db`: projects, datasets and their files. A dataset is added marked as being written, and listed once it is finished. Each connection to the shared file waits up to 5 s for another's write.
5. **Nothing half written is listed:**
   - a writer dropped before it finishes removes its folder and its row;
   - a dataset still marked when the store opens, left by a crash, is removed then.
6. **A dataset's size is known when it ends:** a writer plans a number of frames, and an array holding fewer is cut to them when it finishes.

## Consequences

- **Gains:**
  - napari, Fiji and the OME tools can open a dataset without Tetera;
  - a dataset is findable by its project and name, and its folder is self-describing;
  - a crash or a cancel leaves no dataset that isn't whole.
- **Costs:**
  - a chunk a frame makes many files for long time series;
  - `zarrs` brings about seventy crates, zstd's C library among them;
  - the thumbnail is a quick look, not the preview's colour.
- **To watch:**
  - pyramids and tiles, for scans' mosaics (P1.17), as further arrays in the multiscales;
  - sharding, if many small chunks slow the filesystem;
  - checking against OME-NGFF's own JSON schemas, rather than the tests' reading of the spec.

## Alternatives considered

- **OME-TIFF:** widely read, but one file grows for the whole dataset, so a crash or a cancel spoils it, and appending frames rewrites its metadata.
- **HDF5:** a single file with the same fault, and a C library with a long history of corruption on crashes.
- **Zarr v2 and OME-NGFF 0.4:** read more widely today, but 0.5 is the current spec and v3 the format going forward.
- **Folders without an index:** nothing to query, and a half-written dataset would look like a whole one.
