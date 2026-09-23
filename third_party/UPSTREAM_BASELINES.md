# Pinned upstream baselines

`third_party/verl` and `third_party/edgerazor` are submodules of the public
upstream repositories, pinned to the commits below. Our changes to them are not
committed into the submodules; they live as patches in `patches/` and are
applied by `setup/bootstrap.sh`.

| Dependency | Upstream | Pinned commit | Date |
|---|---|---|---|
| verl | https://github.com/volcengine/verl | `bdfedd4617931e35f05c17c2c4954489efe68054` | 2026-06-25 |
| EdgeRazor | https://github.com/zhangsq-nju/EdgeRazor | `f48a29987d737f2f83f13a58245ee055d300b2b7` | 2026-06-04 |

Both are Apache-2.0.

## How these commits were chosen

The work was done against vendored copies whose recorded upstream commits
(`b951198f` for verl, `f9757655` for EdgeRazor) are not reachable from the
public remotes — they came from a mirror that is not public. Verified against
full clones of both histories.

Each pinned commit above is therefore the public commit whose tree is closest
to the vendored copy, found by content fingerprinting: a 64-commit window for
verl and a full history scan for EdgeRazor. The patches are the diff from that
commit to our tree, so a small part of each may be upstream drift rather than a
change of ours.

`__pycache__`, `*.pyc` and two patch-leftover files
(`verl/trainer/ppo/v1/trainer_base.py.orig`,
`tests/experimental/agent_loop/test_teacher_forcing_agent_loop_on_cpu.py.orig`)
are excluded from the patches.

## Re-applying by hand

```bash
git -C third_party/verl apply --check third_party/patches/verl-qaopd.patch
```

`bootstrap.sh` runs this check before applying and is safe to re-run: it
detects an already-patched tree with `git apply --check -R`.

## Do not clone with `--recursive`

EdgeRazor records `example/medical_vit/vit.cpp` as a gitlink but ships no
`.gitmodules` entry for it, so recursing aborts the whole submodule update and
leaves verl on its default branch instead of the pinned commit. Let
`bootstrap.sh` fetch the submodules.
