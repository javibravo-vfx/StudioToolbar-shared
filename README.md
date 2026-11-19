# Studio Toolbar — shared

DCC-agnostic data and code read by every Studio Toolbar (3ds Max, Houdini, Blender, Maya).
Deployed at `T:\.studio-toolbar\shared\`, next to `T:\.studio-toolbar\max\` and the others.

```
python/studio_lookdev/   core.py (paths, presets, sidecars, framing / layout / colour math)
                         library.py (thumbnails, blurred backgrounds, .tx — image I/O given by each DCC)
lookdev/presets/         turntable presets (.json)
lookdev/reference/       colour chart definitions (.json, CIELAB D50) — _TEMPLATE_chart.json to add one
lookdev/hdri/            HDRIs + optional name.json sidecar  (not in git)
lookdev/_library_cache/  previews of other HDRI folders     (generated, not in git)
```

HDRIs and generated previews are not versioned: they live only on the studio drive.
`STUDIO_LOOKDEV_DIR` points the lookdev folder elsewhere.
