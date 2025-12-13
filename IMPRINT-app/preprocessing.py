import numpy as np
import pandas as pd

def prepare_input(age, aub, pain, abdominal_distress,
                  diameter_mm, margins_irregular,
                  cystic_areas, cs,
                  platelets, neutrophils, lymphocytes,
                  monocytes, eosinophils,
                  transforms=None):

    # Raw SIMs
    NLR = neutrophils / max(lymphocytes, 1e-9)
    SII = platelets * neutrophils / max(lymphocytes, 1e-9)
    PIV = platelets * neutrophils * monocytes / max(lymphocytes, 1e-9)

    row = {
        "Menopausal_status": 0,  # set later in app
        "Margins_bin": int(bool(margins_irregular)),
        "Cystic_areas": int(bool(cystic_areas)),
        "CS": float(cs),
        "Max diameter (mm)": float(diameter_mm),
        "NLR": float(NLR),
        "SII": float(SII),
        "PIV": float(PIV),
    }

    def apply_z(val, mu, sd):
        return (val - mu) / sd if np.isfinite(sd) and sd != 0 else np.nan

    def maybe_log_fixed(val, flag):
        return np.log(val) if flag and val > 0 else val

    if transforms:
        row["_cs_z"] = apply_z(row["CS"], transforms["CS"]["mu"], transforms["CS"]["sd"])
        row["_diam_z"] = apply_z(row["Max diameter (mm)"], transforms["Max diameter (mm)"]["mu"], transforms["Max diameter (mm)"]["sd"])
        for sim in ["NLR","SII","PIV"]:
            v_t = maybe_log_fixed(row[sim], transforms[sim]["log"])
            row[f"_{sim}_z"] = apply_z(v_t, transforms[sim]["mu"], transforms[sim]["sd"])

    return pd.DataFrame([row])