# Change log

Best viewed [here](https://qstheory.github.io/matterwave/main/changelog.html).

## Unreleased

## Matterwave 0.5.2 (04 December 2025)
- Fixed that `plot_array` and `generate_panel_plot` plotted the position space values instead of the frequency space values in the frequency space plot.
- Fixed potential propagator in `split_step` which effectively led to propagation with 2*V

## Matterwave 0.5.1 (03 September 2025)
- `get_ground_state_ho` has now a `device` argument to control on which device the `Array` is created.
- Rename parameter `m` to `mass` in `get_e_kin` in order to be more consistent with the other functions.

## Matterwave 0.5.0 (17 July 2025)
- Initial public release.
