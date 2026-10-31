# Data pengujian lapangan

Folder ini berisi berkas pengukuran nyata yang dikumpulkan peneliti. Berkas yang tersedia hanyalah template
kosong; jangan diisi dengan nilai karangan.

- `distance_measurements_template.csv`: salin menjadi `distance_measurements.csv`. Satu baris per objek terukur:
  `image,x1,y1,x2,y2,gt_distance_m,object_name,notes`.
- `navigation_scenarios_template.csv`: salin menjadi `navigation_scenarios.csv`. Satu baris per skenario:
  `scenario_id,image,gt_direction,description`.

Citra disimpan di folder ini (atau subfolder) dan dirujuk dengan path relatif. Protokol lengkap ada di
`docs/EXPERIMENTS.md` §4–5.
