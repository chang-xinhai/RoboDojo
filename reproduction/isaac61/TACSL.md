# RoboDojo TacSL integration

`env/tactile_manager` uses the official `isaaclab_contrib.sensors.tacsl_sensor.VisuoTactileSensor` implementation from Isaac Lab 3.0 Early Access. It loads the original NVIDIA GelSight R1.5 USD, camera, gel meshes and Taxim calibration; there is no replacement tactile image generator. A RoboDojo-owned subclass adapts the camera buffer boundary (`ProxyArray.torch` and legacy/canonical depth keys), single-environment buffer allocation, exact PhysX view paths and native SDF coordinate units, retaining the official force equations and Taxim methods.

Each ARX X5 arm has two sensors, on `link7` and `link8`. The native gel points toward −Y; the second finger is rotated 180° around Z so its gel faces the opposing jaw. A fixed joint attaches the sensor to the moving finger. The sensor's original world joint and redundant housing meshes are disabled; its compliant gel collider remains active. The original camera geometry and calibration resolution remain 240 pixels wide × 320 high.

The X5 mounting jaw's backing lies 16.53 mm in front of the internal sensor camera. Its 17.3 mm near clipping plane excludes that backing; the original 18.6 mm far plane bounds the gel/contact imaging volume. Camera intrinsics and the official Taxim calibration remain unchanged. Calibration rejects empty RTX buffers and warms actual rendering while physics remains frozen.

The RGB image comes from official Taxim rendering of the actual RTX depth change. `depth` is absolute camera distance in metres, not indentation. `penetration_depth` is SDF indentation in metres. `normal_force` and `shear_force` are official TacSL penalty-model estimates in newtons, not an external measured load cell. The latter has two tangent components per taxel. The default grid has 12 × 16 taxels.

Each rigid mesh collider, including every collider of articulated task objects, gets a separate official SDF view. Forces add across collider views; penetration records the maximum indentation. Exact mesh paths prevent official TacSL's first-child selection from omitting subsequent colliders. The manager prepares colliders before PhysX initialization and preserves references across pose resets. Cloth-only scenes supply actual tactile RGB/depth and declare the four force-related fields `None`, with `force_field_available: false` and an explicit reason. They do not generate zero arrays and label them valid force.

Native PhysX SDF queries use unscaled mesh coordinates, while the official TacSL contact method supplies body coordinates in metres. `sdf_boundary.BodyFrameSdfView` converts points through the actual USD mesh-to-body translation, rotation and positive uniform geometry scale, queries the real native SDF, rotates its gradient back to the body frame and scales its distance back to metres. This supports child collider offsets and rotations, including uniformly scaled parent bodies. Nonuniform or nonpositive scales fail explicitly. The wrapper does not change the TacSL contact law or synthesize distances. New collider topology after initialization requires a fresh TaskEnv; capture detects a changed topology and raises instead of silently dropping contacts.

PhysX particle tasks require host readback, whose native CPU tensor view does not implement SDF shape queries. Only this path uses `WarpTriangleMeshSdfView`: it builds a real Warp mesh from each USD collider's actual triangle vertices and indices, determines inside/outside with the winding-number query and returns signed closest-surface distances and gradients. Body coordinates, scale conversion and the official TacSL penalty kernels remain the same. `provenance.contact_query_backend` explicitly reports `warp_triangle_mesh_signed_distance`; normal tasks report `physx_native_sdf`. Force estimates cover the registered rigid collider surfaces, not fluid pressure. Nontriangulated collider meshes fail explicitly rather than changing their geometry silently.

Exactly zero-area faces are excluded because they have no physical surface or normal; a collider containing only such faces fails. Readback provenance records the remaining triangle count and discarded zero-area count for each collider. Nonzero triangles are retained without an arbitrary size cutoff.

Different parallel environments may contain different objects and collider counts. Each environment has four independent camera sensors with exact body/SDF paths; outputs stack into the usual environment batch. This costs four RTX sensor render products per environment. Images and forces are acquired from their actual environment; mixed cloth/rigid batches retain per-environment availability metadata. Explicit `contact_object_paths` accept `{ENV_REGEX_NS}`, the usual `env_.*`/`env_.+`/`env_[^/]+` forms or the source `env_0` namespace, which are resolved to exact paths before sensor construction. Unknown expressions and paths into another environment fail clearly.

Acquisition updates the official sensors while the simulation physics frame stays fixed. Each finger includes `timestamp_ns`, `physics_frame`, acquisition semantics and backend provenance. The observation manager checks the same physics clock around RGB, depth, stereo IR and tactile acquisition.

Download the exact sensor assets with:

```bash
bash reproduction/isaac61/run.sh python reproduction/isaac61/fetch_tacsl_assets.py
```

The downloader checks pinned SHA256 hashes and writes `Assets/TacSL/source_manifest.json`. NVIDIA publishes this sensor USD/calibration asset under the 5.1 asset bucket; the runtime and API used here are Isaac Sim 6.1 / Isaac Lab 3.0 EA. These asset files are not a second simulator installation.

The physical four-finger acceptance test is:

```bash
bash reproduction/isaac61/run.sh tactile-smoke
bash reproduction/isaac61/run.sh tactile-smoke --num_envs 2 --output reproduction/isaac61/outputs/tactile-contact-env2
# Exercise particle host readback and the actual Warp triangle-mesh backend:
bash reproduction/isaac61/run.sh tactile-smoke --num_envs 2 --particle-readback --output reproduction/isaac61/outputs/tactile-contact-readback-env2
```

It moves an actual PhysX SDF mesh sphere into all four mounted gel pads in each environment, asserts positive indentation and force plus changed tactile RGB, and saves the measured arrays, JSON metrics and a before/contact gallery under the output directory. The parallel test also asserts that another environment's force stays zero during each isolated probe. A failed assertion exits nonzero. A run is accepted only after `result.json` reports `status: passed` for every finger and environment.

## Verified Isaac Sim 6.1 result

The two-environment test passed on Isaac Sim distribution **6.1.0.0** with Isaac Lab **17.0.2**, using the pinned 3.0 EA source `ae37b028ea415c91ea2bc32609efcd759ed2b974`. The second environment uses a collider rotated 45° about Z, a 20 mm child translation and a 1.25 uniform body scale, giving a physical 25 mm mesh-to-body offset. All eight mounted fingers produced real contact optical changes, positive SDF indentation and positive TacSL normal force. Cross-environment isolation assertions passed.

| Measurement | Observed range |
| --- | --- |
| Maximum normal force per finger | 0.0528–0.1019 N |
| Maximum indentation per finger | 0.317–0.433 mm |
| Mean absolute tactile RGB change, 0–255 scale | 2.528–2.656 |
| Native SDF error at known ±1 mm physical offsets | Less than 0.039 mm |

The scaled environment's actual native query distances were +0.999981 mm and −0.961885 mm at the known ±1 mm points; this validates the conversion to physical metres against native GPU queries. The remaining difference comes from the sampled SDF representation. These results are from physical probes against real mounted sensors; the force values remain estimates from the official penalty model.

Artifacts are in [result.json](outputs/tactile-contact-env2/result.json), [gallery.png](outputs/tactile-contact-env2/gallery.png) and the eight per-finger NPZ files under `outputs/tactile-contact-env2`. The gallery's left image is the actual no-contact render; the right image is the actual contact render. The smoke script saves RGB, depth, baseline RGB/depth and force arrays and exits nonzero on exceptions, including exceptions during shutdown.

The particle host-readback variant also passed on the same exact runtime, with all eight actual contact probes and cross-environment isolation assertions. Its explicitly labelled Warp triangle-mesh backend produced 0.0410–0.1049 N maximum normal force, 0.319–0.433 mm indentation and 2.526–2.656 mean absolute RGB change. Known physical ±1 mm queries returned +1.000000 mm and −0.995747 mm, an error below 4.3 µm from the triangulated sphere surface. Its [result.json](outputs/tactile-contact-readback-env2/result.json), [gallery.png](outputs/tactile-contact-readback-env2/gallery.png) and NPZ files are separate from the native PhysX SDF results.
