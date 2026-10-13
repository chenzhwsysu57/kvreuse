# Third-party code and data

This repository does not relicense third-party code or datasets.

## RelayCaching

- Source: https://github.com/YingshengGeng/RelayCaching
- Base revision: `6b0e9d37b74d89cfb0fbad276790150a0d9c0b49`.
- License: the upstream MIT license is retained at
  [third_party/RelayCaching/LICENSE](third_party/RelayCaching/LICENSE).
- Only Python source, upstream README, requirements and license are vendored;
  nested Git history, upstream data, model files and experiment artifacts are excluded.
- `main` preserves the local source checkout, including its existing generation
  timing instrumentation. `batch` adds the two-file batching change from revision
  `18e2d061ba9118dd0a5905bee77e400d7b0e7aec` and the corresponding runner.
- The vendor directory is ordinary Git-tracked source, not a submodule. No
  additional upstream checkout or unpublished submodule commit is required.
- The upstream requirements are retained for provenance only. Use the minimal
  [requirements-relay.txt](requirements-relay.txt) in a separate environment.

## Evaluation datasets

See [data/README.md](data/README.md). Deal data is **CC BY-NC 4.0**, not MIT.
ArgKP's upstream README does not specify a complete standalone dataset license;
check IBM's applicable terms before redistribution or commercial use. Original
upstream notices are retained with the data. No upstream license is inferred from
the code license, and this repository grants no additional rights to those data.

## First-party licensing

First-party project code is released under the [MIT License](LICENSE).
This license does not override the licenses or usage restrictions of third-party
code, datasets, or model weights.