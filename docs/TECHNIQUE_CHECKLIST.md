# Technique checklist (template - requires clinical review)

The classifier is only as valid as these definitions. Fill this in for the **specific device** used
in the project (pMDI with/without spacer, DPI, ...), have a healthcare professional review and sign
it off, and only then label recordings. Do not ship numbers from this template.

| Item | Value |
|---|---|
| Device / model | _e.g. pMDI + valved holding chamber_ |
| Reference guideline | _cite the manufacturer / clinical guideline used_ |
| Reviewed by (name, role, date) | |

## Visible criteria

| # | Criterion (camera-observable) | Class if violated | Numeric definition (fill in) | Feature used |
|---|---|---|---|---|
| 1 | Mouthpiece is at / inside the mouth | Wrong Mouth Position | `mp_mouth_dist` <= ___ face-widths | `mp_mouth_dist` |
| 2 | Lips are around the mouthpiece (mouth open to receive it) | Wrong Mouth Position | `mouth_open_ratio` >= ___ | `mouth_open_ratio` |
| 3 | Inhaler held in the orientation required by the device | Wrong Inhaler Position | angle vs head within ___ +- ___ deg | `inhaler_angle_rel_head` |
| 4 | Mouthpiece points toward the mouth | Wrong Inhaler Position | `mp_rel` direction within ___ deg of mouth axis | `mp_rel_x/y` |
| 5 | Head approximately upright / facing camera | Unable to Determine if violated | yaw <= ___ deg, pitch <= ___ deg | `head_*` |

## Not assessable with an RGB camera (must be stated in any report)

Airflow / inspiratory flow, breath-hold, actuation timing, dose delivery, inhalation quality, device
shake before use, rinsing the mouth afterwards.

## Labelling rules

* One label per recording segment: `correct`, `wrong_mouth`, `wrong_inhaler`.
* If both are wrong, label the dominant error and note the other in `error_type` (or move to the
  multi-label schema).
* Record the intended error (`error_type`, e.g. `mouth_too_far`, `inhaler_tilted`) at capture time
  rather than judging afterwards.
* Two independent raters on a 10-20% sample; report agreement (Cohen's kappa).
