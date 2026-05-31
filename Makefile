# vid2scene — one-command DX for the laptop-runnable (CPU) stages.
# The GPU reconstruction stage is driven by scripts/remote/ (see docs/ARCHITECTURE.md).

.PHONY: help install ingest fuse benchmark viz embodied viewer demo clean

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:         ## pip install the package (CPU stages)
	pip install -e .

test:            ## run the unit tests
	pytest tests/ -q

ingest:          ## select keyframes from a video: make ingest VIDEO=clip.mp4 OUT=runs/ing
	vid2scene ingest $(VIDEO) --out $(OUT)

fuse:            ## consensus-fuse two meshes: make fuse BACKBONE=pgsr.ply DONOR=dn.ply OUT=cons.ply
	vid2scene fuse --backbone $(BACKBONE) --donor $(DONOR) --out $(OUT)

benchmark:       ## reproduce the GT-mesh benchmark table from the bundled results
	vid2scene benchmark --eval-dir runs/replica_eval

viz:             ## decimate a mesh for the web viewer: make viz IN=mesh.ply OUT=web.ply
	vid2scene viz --in $(IN) --out $(OUT)

embodied:        ## export a sim-ready GLB: make embodied MESH=room.ply OUT=room_sim.glb
	vid2scene embodied --mesh $(MESH) --out $(OUT) --scene-json $(OUT).json

viewer:          ## serve the interactive web viewers at http://localhost:8765
	python3 -m http.server 8765

demo:            ## print the headline result + viewer URLs (no GPU needed)
	@echo "=== vid2scene: phone scan -> metric 3D -> robot-explorable world ==="
	@$(MAKE) -s benchmark
	@echo "\nInteractive viewers (run 'make viewer' then open):"
	@echo "  http://localhost:8765/viewer/replica_room0.html   (benchmark: recon vs GT)"
	@echo "  http://localhost:8765/viewer/mesh_compare.html     (4 methods + consensus)"
	@echo "  http://localhost:8765/viewer/splat_ref.html        (reference Gaussian splat)"

clean:           ## remove caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} + ; rm -rf build *.egg-info
