Model	Input	Time	Mesh Output	Key Advantage
InstantMesh	1-4 images (generates multi-view if given 1)	~10-30s	Yes, .obj	Multi-view aware, open source
Trellis (Microsoft)	1-4 images	~10-15s	Yes, .glb/.obj	High quality, handles diverse objects
CRM (Convolutional Reconstruction Model)	6 canonical views	~10s	Yes, .obj	Explicit multi-view, no guessing
Wonder3D	1 image → generates 6 views → reconstructs	~2-3 min	Yes, .obj	Cross-domain diffusion for novel views
Era3D	1 image → generates multi-view → reconstructs	~1-2 min	Yes, .obj	Better multi-view consistency
COLMAP (SfM)	Multi-view (10-50+ images)	Very slow (often dominates runtime; scales ~O(n²) matching)	Sparse point cloud + cameras, not a mesh	Industry-standard poses; bottleneck before splat/NeRF training
3D Gaussian Splatting + SuGaR/2DGS	Multi-view (10-50 images)	5-15 min total (plus COLMAP time above)	Yes (mesh extraction)	Uses ALL your photos, high fidelity
