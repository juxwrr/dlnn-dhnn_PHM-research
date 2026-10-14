# Hail Mary physics - complete deterministic package

This folder contains the complete work requested in `source/project_roadmap.txt`. Start with `docs/model_specification.md` for the frozen equations, `data/numerical_results.csv` for headline values, and `data/scenario_summary.csv` for the ten reference cases. The corrected workbook is `astrophage_physics_corrected.xlsx`. Source code is in `src/`; every plot is reproducible by running `python3 src/generate_all.py` from this folder, followed by the workbook builder.

Headline corrected results:

- Half-trip rapidity: 3.03213934; peak beta: 0.99536191; peak gamma: 10.39488544.
- Ideal trip: 3.840649 crew years and 13.105579 Earth years.
- Correct accelerate/brake mass ratio: 3620692.2; fuel: 3.6206912e+11 kg; volume: 71301521 m^3.
- Corrected Beetle peak beta: 0.61998581; Earth trip: 19.194417 years.
- Corrected navigation deceleration: 15.67893051 m/s^2; stop time: 8.770705 days.

Generated 48 Matplotlib PNG figures. See `docs/plot_catalog.md` for the complete list.
