import re  
import json
text='''(AGNIFingerPrint_std_dev_AGNI_dir=x_eta=1_88e+00 - BondFractions_O_-Sn_bond_frac) + (DensityFeatures_packing_fraction * CrystalNNFingerprint_mean_wt_CN_2)
(AGNIFingerPrint_std_dev_AGNI_eta=1_88e+00 - CrystalNNFingerprint_mean_linear_CN_2) / (|StructuralHeterogeneity_max_relative_bond_length - ElementProperty_MagpieData_mean_NdValence|)
(AGNIFingerPrint_std_dev_AGNI_eta=1_88e+00 - CrystalNNFingerprint_std_dev_wt_CN_2) / (|StructuralHeterogeneity_max_relative_bond_length - ElementProperty_MagpieData_mean_NdValence|)
(AGNIFingerPrint_std_dev_AGNI_eta=4_43e+00 * ElementProperty_MagpieData_minimum_MendeleevNumber) * (ElementProperty_MagpieData_minimum_Electronegativity - ElementProperty_MagpieData_mean_NUnfilled)
(AtomicOrbitals_HOMO_element * CrystalNNFingerprint_std_dev_wt_CN_2) * (ElementProperty_MagpieData_maximum_GSvolume_pa / CoulombMatrix_coulomb_matrix_eig_0)
(AtomicPackingEfficiency_dist_from_5_clusters__APE____0_010⁶) / (ValenceOrbital_frac_d_valence_electrons - DensityFeatures_packing_fraction)
(BandCenter_band_center * CoulombMatrix_coulomb_matrix_eig_2) + (VoronoiFingerprint_mean_Voro_dist_minimum⁶)
(BondOrientationParameter_mean_BOOP_Q_l=4⁶) / (|DensityFeatures_packing_fraction - ElementProperty_MagpieData_minimum_Electronegativity|)
(BondOrientationParameter_std_dev_BOOP_Q_l=1 * ElementFraction_N) * (VoronoiFingerprint_mean_Voro_area_minimum²)
(CoulombMatrix_coulomb_matrix_eig_1 * ElementProperty_MagpieData_avg_dev_NUnfilled) / (ElementProperty_MagpieData_minimum_Electronegativity - ElementProperty_MagpieData_mean_NdUnfilled)
(CrystalNNFingerprint_mean_linear_CN_2 * SineCoulombMatrix_sine_coulomb_matrix_eig_0) * (ElementProperty_MagpieData_maximum_MeltingT * ElementProperty_MagpieData_avg_dev_GSbandgap)
(CrystalNNFingerprint_mean_wt_CN_5 * SineCoulombMatrix_sine_coulomb_matrix_eig_1) * (DensityFeatures_packing_fraction * ElementProperty_MagpieData_maximum_NdUnfilled)
(CrystalNNFingerprint_mean_wt_CN_6⁶) / (ValenceOrbital_frac_f_valence_electrons - BondOrientationParameter_mean_BOOP_Q_l=4)
(CrystalNNFingerprint_std_dev_hexagonal_planar_CN_6 * BondFractions_Ag_-O_bond_frac) + (ℓ-OFM_v1_177 * GaussianSymmFunc_std_dev_G2_80_0)
(CrystalNNFingerprint_std_dev_hexagonal_planar_CN_6 / ElementProperty_MagpieData_minimum_MendeleevNumber) - (XRDPowderPattern_xrd_37 / ElementProperty_MagpieData_Data_minimum_Electronegativity)
(CrystalNNFingerprint_std_dev_hexagonal_planar_CN_6 / ElementProperty_MagpieData_minimum_MendeleevNumber) - (XRDPowderPattern_xrd_37 / ElementProperty_MagpieData_minimum_Column)
(CrystalNNFingerprint_std_dev_hexagonal_planar_CN_6 / ElementProperty_MagpieData_minimum_MendeleevNumber) - (XRDPowderPattern_xrd_37 / ElementProperty_MagpieData_minimum_Electronegativity)
(CrystalNNFingerprint_std_dev_linear_CN_2 + ValenceOrbital_frac_f_valence_electrons) / (|DensityFeatures_packing_fraction - ElementProperty_MagpieData_minimum_Electronegativity|)
(CrystalNNFingerprint_std_dev_pentagonal_pyramidal_CN_6 * ElementFraction_N) / (ElementProperty_MagpieData_mean_Electronegativity - GlobalSymmetryFeatures_crystal_system)
(CrystalNNFingerprint_std_dev_trigonal_pyramidal_CN_4 / ElementProperty_MagpieData_mode_Electronegativity) / (ElementProperty_MagpieData_mode_Electronegativity + ℓ-OFM_v1_171)
(CrystalNNFingerprint_std_dev_wt_CN_2 * ElectronegativityDiff_range_EN_difference) / (ElementProperty_MagpieData_mean_Column - ElementProperty_MagpieData_range_GSbandgap)
(DensityFeatures_packing_fraction * CrystalNNFingerprint_mean_wt_CN_2) + sqrt(BondOrientationParameter_mean_BOOP_Q_l=1)
(DensityFeatures_packing_fraction - CrystalNNFingerprint_std_dev_wt_CN_2) / (ElementProperty_MagpieData_avg_dev_SpaceGroupNumber + ElementProperty_MagpieData_maximum_GSvolume_pa)
(ElectronegativityDiff_minimum_EN_difference * ElectronegativityDiff_range_EN_difference) * (CrystalNNFingerprint_mean_linear_CN_2 - ElementProperty_MagpieData_minimum_Electronegativity)
(ElectronegativityDiff_minimum_EN_difference * ElectronegativityDiff_range_EN_difference) * (ElementProperty_MagpieData_minimum_Electronegativity - CrystalNNFingerprint_std_dev_wt_CN_2)
(ElementFraction_B / AGNIFingerPrint_std_dev_AGNI_eta=8_00e-01) + (ElementProperty_MagpieData_avg_dev_MeltingT / ElementProperty_MagpieData_minimum_GSvolume_pa)
(ElementProperty_MagpieData_avg_dev_GSbandgap * ElementProperty_MagpieData_maximum_NUnfilled) * (ElementProperty_MagpieData_maximum_NdUnfilled / CrystalNNFingerprint_std_dev_wt_CN_2)
(ElementProperty_MagpieData_avg_dev_GSbandgap / CrystalNNFingerprint_std_dev_wt_CN_2) * (ElementProperty_MagpieData_maximum_NdUnfilled³)
(ElementProperty_MagpieData_avg_dev_MeltingT * ElementProperty_MagpieData_maximum_GSvolume_pa) * (CrystalNNFingerprint_mean_linear_CN_2 / DensityFeatures_density)
(ElementProperty_MagpieData_avg_dev_MeltingT / ElementProperty_MagpieData_avg_dev_CovalentRadius) + (BondOrientationParameter_mean_BOOP_Q_l=4 * ElementProperty_MagpieData_maximum_GSvolume_pa)
(ElementProperty_MagpieData_maximum_CovalentRadius * ElementProperty_MagpieData_avg_dev_GSbandgap) * (ElementProperty_MagpieData_maximum_NdUnfilled / CrystalNNFingerprint_std_dev_wt_CN_2)
(ElementProperty_MagpieData_maximum_GSvolume_pa / CrystalNNFingerprint_mean_pentagonal_pyramidal_CN_6) / (|ElectronegativityDiff_range_EN_difference - ElementProperty_MagpieData_minimum_MendeleevNumber|)
(ElementProperty_MagpieData_maximum_MeltingT * GaussianSymmFunc_std_dev_G2_80_0) - (ElementProperty_MagpieData_minimum_Electronegativity * ElementProperty_MagpieData_maximum_NUnfilled)
(ElementProperty_MagpieData_maximum_MeltingT * GaussianSymmFunc_std_dev_G2_80_0) - (ℓ-OFM_v1_72 + ElementProperty_MagpieData_maximum_NUnfilled)
(ElementProperty_MagpieData_maximum_MeltingT * GaussianSymmFunc_std_dev_G2_80_0) - (ℓ-OFM_v1_72 + ElementProperty_MagpieData_mean_NUnfilled)
(ElementProperty_MagpieData_maximum_MeltingT / DensityFeatures_density) / (AGNIFingerPrint_std_dev_AGNI_dir=x_eta=1_88e+00 + CrystalNNFingerprint_std_dev_wt_CN_2)
(ElementProperty_MagpieData_maximum_NdUnfilled / CrystalNNFingerprint_std_dev_wt_CN_2) / (exp(-1.0 * ElementProperty_MagpieData_avg_dev_GSbandgap))
(ElementProperty_MagpieData_maximum_NdValence * DensityFeatures_packing_fraction) + (|ℓ-OFM_v1_72 - ElementProperty_MagpieData_maximum_NUnfilled|)
(ElementProperty_MagpieData_maximum_NdValence / AGNIFingerPrint_std_dev_AGNI_eta=8_00e-01) / (ElementProperty_MagpieData_maximum_NdUnfilled - VoronoiFingerprint_mean_Voro_dist_minimum)
(ElementProperty_MagpieData_mean_Column * CoulombMatrix_coulomb_matrix_eig_2) + (ElementProperty_MagpieData_mean_NpValence * ElementProperty_MagpieData_mean_MeltingT)
(ElementProperty_MagpieData_mean_NdValence / DensityFeatures_packing_fraction) / (|MaximumPackingEfficiency_max_packing_efficiency - ElementProperty_MagpieData_mean_NdValence|)
(ElementProperty_MagpieData_mean_SpaceGroupNumber / ElementProperty_MagpieData_mean_NpValence) / (ElementProperty_MagpieData_mean_NpValence + ElementProperty_MagpieData_maximum_NdUnfilled)
(ElementProperty_MagpieData_mean_SpaceGroupNumber / VoronoiFingerprint_mean_Voro_area_sum) / (ElementProperty_MagpieData_mean_NpValence + ElementProperty_MagpieData_maximum_NdUnfilled)
(ElementProperty_MagpieData_minimum_NsValence⁶) / (CoulombMatrix_coulomb_matrix_eig_2 * AGNIFingerPrint_std_dev_AGNI_eta=8_00e-01)
(ElementProperty_MagpieData_range_CovalentRadius / DensityFeatures_density) / (|CrystalNNFingerprint_std_dev_linear_CN_2 - DensityFeatures_packing_fraction|)
(ElementProperty_MagpieData_range_Electronegativity * ElementProperty_MagpieData_maximum_MeltingT) * (ValenceOrbital_frac_d_valence_electrons - CrystalNNFingerprint_mean_linear_CN_2)
(ElementProperty_MagpieData_range_Electronegativity * ElementProperty_MagpieData_range_MeltingT) * (ValenceOrbital_frac_d_valence_electrons - CrystalNNFingerprint_mean_linear_CN_2)
(ElementProperty_MagpieData_range_MeltingT * ElementProperty_MagpieData_avg_dev_GSbandgap) * (ElementProperty_MagpieData_maximum_GSvolume_pa * CrystalNNFingerprint_std_dev_wt_CN_2)
(ElementProperty_MagpieData_range_MeltingT / DensityFeatures_density) / (AGNIFingerPrint_std_dev_AGNI_dir=x_eta=1_88e+00 + CrystalNNFingerprint_std_dev_wt_CN_2)
(ElementProperty_MagpieData_range_MeltingT²) * (XRDPowderPattern_xrd_37 * CrystalNNFingerprint_mean_linear_CN_2)
(GlobalSymmetryFeatures_spacegroup_num / ℓ-OFM_v1_57) / (AverageBondAngle_mean_Average_bond_angle - ℓ-OFM_v1_172)
(SineCoulombMatrix_sine_coulomb_matrix_eig_2²) / (ℓ-OFM_v1_83 + ℓ-OFM_v1_57)
(StructuralHeterogeneity_maximum_neighbor_distance_variation - YangSolidSolution_Yang_delta) / (|ElementProperty_MagpieData_mean_NValence - ElementProperty_MagpieData_range_GSbandgap|)
(ValenceOrbital_frac_d_valence_electrons - CrystalNNFingerprint_mean_linear_CN_2) * (ElementProperty_MagpieData_maximum_MeltingT / ElementProperty_MagpieData_minimum_Electronegativity)
(VoronoiFingerprint_mean_Voro_area_minimum * ElementFraction_N) - (CrystalNNFingerprint_std_dev_wt_CN_2 / ElementProperty_MagpieData_minimum_Column)
(VoronoiFingerprint_mean_Voro_dist_minimum - ElementProperty_MagpieData_mean_Electronegativity) * (ElectronegativityDiff_minimum_EN_difference * ElectronegativityDiff_range_EN_difference)
(VoronoiFingerprint_mean_Voro_dist_minimum⁶) / (|VoronoiFingerprint_mean_Voro_dist_minimum - ElementProperty_MagpieData_minimum_MendeleevNumber|)
(VoronoiFingerprint_std_dev_Voro_area_sum⁶) / (ElementProperty_MagpieData_mode_Electronegativity * GlobalSymmetryFeatures_spacegroup_num)
(VoronoiFingerprint_std_dev_Voro_area_sum⁶) / ln(CrystalNNFingerprint_mean_L-shaped_CN_2)
(VoronoiFingerprint_std_dev_Voro_vol_sum / CrystalNNFingerprint_mean_hexagonal_planar_CN_6) / (|DensityFeatures_packing_fraction - ElementProperty_MagpieData_minimum_Electronegativity|)
(VoronoiFingerprint_std_dev_Voro_vol_sum / ElementProperty_MagpieData_minimum_Column) + (ElementProperty_MagpieData_mean_MeltingT / ElementProperty_MagpieData_maximum_CovalentRadius)
(XRDPowderPattern_xrd_41 * ElementProperty_MagpieData_range_CovalentRadius) / (|CrystalNNFingerprint_std_dev_linear_CN_2 - DensityFeatures_packing_fraction|)
(|ElementProperty_MagpieData_mean_MeltingT - ElementProperty_MagpieData_avg_dev_MeltingT|) * (XRDPowderPattern_xrd_37 * CrystalNNFingerprint_mean_linear_CN_2)
(|ElementProperty_MagpieData_mean_NdUnfilled - CrystalNNFingerprint_mean_linear_CN_2|) / (|CrystalNNFingerprint_mean_octahedral_CN_6 - DensityFeatures_packing_fraction|)
(|ElementProperty_MagpieData_mean_SpaceGroupNumber - VoronoiFingerprint_mean_Voro_area_sum|) / (ElementProperty_MagpieData_mean_NpValence + ElementProperty_MagpieData_maximum_NdUnfilled)
(|LocalPropertyDifference_std_dev_local_difference_in_Electronegativity - ℓ-OFM_v1_76|) / (|StructuralHeterogeneity_maximum_neighbor_distance_variation - ValenceOrbital_frac_d_valence_electrons|)
(|StructuralHeterogeneity_min_relative_bond_length - DensityFeatures_packing_fraction|) / (ValenceOrbital_frac_f_valence_electrons - ElementProperty_MagpieData_minimum_Electronegativity)
(|ℓ-OFM_v1_97 - ElementFraction_N|) / (CrystalNNFingerprint_std_dev_hexagonal_planar_CN_6 * VoronoiFingerprint_mean_Voro_area_minimum)
(ℓ-OFM_v1_164 + ElectronegativityDiff_range_EN_difference) / (|VoronoiFingerprint_mean_Voro_area_sum - CoulombMatrix_coulomb_matrix_eig_2|)
(ℓ-OFM_v1_76 * AtomicPackingEfficiency_dist_from_5_clusters__APE____0_010) / (|StructuralHeterogeneity_maximum_neighbor_distance_variation - AtomicPackingEfficiency_dist_from_5_clusters__APE____0_010|)
(ℓ-OFM_v1_76 * ElementProperty_MagpieData_maximum_GSvolume_pa) / (|StructuralHeterogeneity_maximum_neighbor_distance_variation - AtomicPackingEfficiency_dist_from_5_clusters__APE____0_010|)
(ℓ-OFM_v1_99 + ElementProperty_MagpieData_maximum_NUnfilled) / (|ElementProperty_MagpieData_avg_dev_SpaceGroupNumber - ElementProperty_MagpieData_minimum_MendeleevNumber|)
exp(CrystalNNFingerprint_mean_wt_CN_3) / (GlobalSymmetryFeatures_spacegroup_num³)
exp(StructuralHeterogeneity_maximum_neighbor_distance_variation) / (GlobalSymmetryFeatures_spacegroup_num³)
|(CrystalNNFingerprint_mean_octahedral_CN_6 * ElementProperty_MagpieData_avg_dev_NUnfilled) - (AverageBondAngle_mean_Average_bond_angle⁶)|
|(ElementProperty_MagpieData_minimum_GSvolume_pa * CrystalNNFingerprint_std_dev_linear_CN_2) - (ElementProperty_MagpieData_avg_dev_NUnfilled * ElementProperty_MagpieData_mean_NdValence)|
|(StructuralHeterogeneity_max_relative_bond_length + VoronoiFingerprint_mean_Voro_dist_minimum) - (DensityFeatures_packing_fraction / AGNIFingerPrint_std_dev_AGNI_eta=1_88e+00)|
|(|ℓ-OFM_v1_68 - ElementProperty_MagpieData_minimum_Column|) - (ℓ-OFM_v1_97 / CrystalNNFingerprint_std_dev_wt_CN_6)|
|ln(ElementProperty_MagpieData_avg_dev_CovalentRadius) - (DensityFeatures_packing_fraction / AGNIFingerPrint_std_dev_AGNI_eta=1_88e+00)|'''
import re
import numpy as np

def convert_formula(expr: str, prefix: str, feature_index: int) -> dict:  
    """  
    Convert a domain-specific feature formula into a pandas/NumPy expression  
    and return it as a dictionary with a SISSO_feature_xx key.  
    - Replaces superscripts (²³⁴⁵⁶) with Python exponent notation.  
    - Transforms |x| into abs(x).  
    - Prepends np. to sqrt, exp, ln.  
    - Wraps variable tokens with df['...'].  
    """  
    # 1. Superscript to Python exponent  
    superscript_map = {  
        '²': '**2', '³': '**3',  
        '⁴': '**4', '⁵': '**5',  
        '⁶': '**6'  
    }  
    for sup, rep in superscript_map.items():  
        expr = expr.replace(sup, rep)  
  
    # 2. Absolute values  
    expr = re.sub(r"\|([^|]+)\|", lambda m: f"abs({m.group(1)})", expr)  
  
    # 3. Math functions → NumPy  
    expr = re.sub(r"\bsqrt\(",    "np.sqrt(", expr)
    expr = re.sub(r"\bcbrt\(",    "np.cbrt(", expr)  
    expr = re.sub(r"\bexp\(",     "np.exp(",  expr)  
    expr = re.sub(r"\bln\(",      "np.log(",  expr)  
  
    # 4. Wrap tokens in df['...']  
    #    Token: starts with letter or ℓ, then word chars, =, -, +, /, ., _  
    token_pattern = re.compile(r"\b[ℓA-Za-z][A-Za-z0-9_\-=+./]*\b")  
  
    def wrap_token(m):  
        tok = m.group(0)  
        # Skip already-numpy or built-in names  
        if tok.startswith("np.") or tok in {  
            'abs','df','np','sin','cos','tan','log','exp','sqrt'  
        }:  
            return tok  
        return f"df['{tok}']"  
  
    converted_term = token_pattern.sub(wrap_token, expr)  
  
    # Return as a dictionary  
    return {f'{prefix}_{feature_index:02d}': converted_term}


# Example usage:  
formulas = [line.strip() for line in text.splitlines() if line.strip()]  
  
converted_formulas_dict = {}  
for i, f in enumerate(formulas):  
    converted_dict = convert_formula(f, 'SISSO_topfeature_coGNperovsk', i + 1)  
    converted_formulas_dict.update(converted_dict)  
  
# Save the dictionary to a JSON file  
output_filename = 'sisso_topfeatures_coGNperovsk.json'  
with open(output_filename, 'w') as json_file:  
    json.dump(converted_formulas_dict, json_file, indent=4)  
  
print(f"Converted formulas saved to {output_filename}")

text_clustered = '''((ValenceOrbital_frac_p_valence_electrons + AverageBondLength_mean_Average_bond_length) - (ℓ-OFM_v1_19 / ElementProperty_MagpieData_maximum_Number))
((ValenceOrbital_frac_p_valence_electrons / ElementProperty_MagpieData_maximum_Number) / (ElectronegativityDiff_minimum_EN_difference + OPSiteFingerprint_mean_sgl_bd_CN_1))
((ElementFraction_O⁶) + cbrt(AverageBondLength_mean_Average_bond_length))
((|ElementFraction_O - ValenceOrbital_frac_d_valence_electrons|) - (ElementProperty_MagpieData_maximum_Electronegativity * ElementFraction_Fe))
((AGNIFingerPrint_std_dev_AGNI_dir=x_eta=1_88e+00 + ElementFraction_O) * (exp(-1.0 * GaussianSymmFunc_mean_G4_0_005_4_0_-1_0)))
(|(AGNIFingerPrint_std_dev_AGNI_dir=x_eta=1_88e+00 + TMetalFraction_transition_metal_fraction) - (ElementFraction_O * ElementProperty_MagpieData_mean_NpUnfilled)|)
((ElementFraction_O * ElementFraction_Fe) - (ValenceOrbital_frac_d_valence_electrons * GeneralizedRDF_mean_Gaussian_center=6_0_width=1_0))
((ElementFraction_O⁶) - (1.0 / AverageBondLength_mean_Average_bond_length))
((OxidationStates_std_dev_oxidation_state + ElementProperty_MagpieData_mean_NpValence) + (MaximumPackingEfficiency_max_packing_efficiency * ℓ-OFM_v1_55))
((|MaximumPackingEfficiency_max_packing_efficiency - ValenceOrbital_frac_p_valence_electrons|) + (MaximumPackingEfficiency_max_packing_efficiency - ValenceOrbital_frac_p_valence_electrons))
((OxidationStates_std_dev_oxidation_state + ElementProperty_MagpieData_mean_NpValence) + (ℓ-OFM_v1_55 / ElementProperty_MagpieData_maximum_Electronegativity))
((|ℓ-OFM_v1_63 - OxidationStates_std_dev_oxidation_state|) - (ℓ-OFM_v1_63 - ℓ-OFM_v1_55))
((|ℓ-OFM_v1_63 - ElementFraction_O|) - (ℓ-OFM_v1_63 - ℓ-OFM_v1_55))
(|(ElementFraction_Yb * OxidationStates_std_dev_oxidation_state) - (ElementProperty_MagpieData_mean_GSbandgap * TMetalFraction_transition_metal_fraction)|)
((|CrystalNNFingerprint_mean_trigonal_non-coplanar_CN_3 - AtomicOrbitals_HOMO_energy|) - cbrt(GeneralizedRDF_std_dev_Gaussian_center=2_0_width=1_0))
((|ℓ-OFM_v1_63 - TMetalFraction_transition_metal_fraction|) - (ℓ-OFM_v1_63 - ℓ-OFM_v1_55))
((ElementProperty_MagpieData_avg_dev_MeltingT * ElementProperty_MagpieData_maximum_GSvolume_pa) * (CrystalNNFingerprint_mean_linear_CN_2 / DensityFeatures_density))
((XRDPowderPattern_xrd_41 * ElementProperty_MagpieData_range_CovalentRadius) / (|CrystalNNFingerprint_std_dev_linear_CN_2 - DensityFeatures_packing_fraction|))
((|LocalPropertyDifference_std_dev_local_difference_in_Electronegativity - ℓ-OFM_v1_76|) / (|StructuralHeterogeneity_maximum_neighbor_distance_variation - ValenceOrbital_frac_d_valence_electrons|))
((ElementProperty_MagpieData_range_CovalentRadius / DensityFeatures_density) / (|CrystalNNFingerprint_std_dev_linear_CN_2 - DensityFeatures_packing_fraction|))
((ℓ-OFM_v1_76 * ElementProperty_MagpieData_maximum_GSvolume_pa) / (|StructuralHeterogeneity_maximum_neighbor_distance_variation - AtomicPackingEfficiency_dist_from_5_clusters__APE____0_010|))
((ℓ-OFM_v1_76 * AtomicPackingEfficiency_dist_from_5_clusters__APE____0_010) / (|StructuralHeterogeneity_maximum_neighbor_distance_variation - AtomicPackingEfficiency_dist_from_5_clusters__APE____0_010|))
((AtomicPackingEfficiency_dist_from_5_clusters__APE____0_010⁶) / (ValenceOrbital_frac_d_valence_electrons - DensityFeatures_packing_fraction))
((ElementProperty_MagpieData_range_Electronegativity * ElementProperty_MagpieData_maximum_MeltingT) * (ValenceOrbital_frac_d_valence_electrons - CrystalNNFingerprint_mean_linear_CN_2))
((CrystalNNFingerprint_mean_octahedral_CN_6 * ElementProperty_MagpieData_avg_dev_NUnfilled) - (AverageBondAngle_mean_Average_bond_angle⁶))
((ElementProperty_MagpieData_range_Electronegativity * ElementProperty_MagpieData_range_MeltingT) * (ValenceOrbital_frac_d_valence_electrons - CrystalNNFingerprint_mean_linear_CN_2))
((CoulombMatrix_coulomb_matrix_eig_1 * ElementProperty_MagpieData_avg_dev_NUnfilled) / (ElementProperty_MagpieData_minimum_Electronegativity - ElementProperty_MagpieData_mean_NdUnfilled))
((|ElementProperty_MagpieData_mean_NdUnfilled - CrystalNNFingerprint_mean_linear_CN_2|) / (|CrystalNNFingerprint_mean_octahedral_CN_6 - DensityFeatures_packing_fraction|))
((ValenceOrbital_frac_d_valence_electrons - CrystalNNFingerprint_mean_linear_CN_2) * (ElementProperty_MagpieData_maximum_MeltingT / ElementProperty_MagpieData_minimum_Electronegativity))
((ElementProperty_MagpieData_mean_NdValence / DensityFeatures_packing_fraction) / (|MaximumPackingEfficiency_max_packing_efficiency - ElementProperty_MagpieData_mean_NdValence|))
(|(ElementProperty_MagpieData_minimum_GSvolume_pa * CrystalNNFingerprint_std_dev_linear_CN_2) - (ElementProperty_MagpieData_avg_dev_NUnfilled * ElementProperty_MagpieData_mean_NdValence)|)
((VoronoiFingerprint_std_dev_Voro_vol_sum / ElementProperty_MagpieData_minimum_Column) + (ElementProperty_MagpieData_mean_MeltingT / ElementProperty_MagpieData_maximum_CovalentRadius))
((CrystalNNFingerprint_std_dev_linear_CN_2 + ValenceOrbital_frac_f_valence_electrons) / (|DensityFeatures_packing_fraction - ElementProperty_MagpieData_minimum_Electronegativity|))
((ElementProperty_MagpieData_range_MeltingT²) * (XRDPowderPattern_xrd_37 * CrystalNNFingerprint_mean_linear_CN_2))
((BondOrientationParameter_mean_BOOP_Q_l=4⁶) / (|DensityFeatures_packing_fraction - ElementProperty_MagpieData_minimum_Electronegativity|))
((|StructuralHeterogeneity_min_relative_bond_length - DensityFeatures_packing_fraction|) / (ValenceOrbital_frac_f_valence_electrons - ElementProperty_MagpieData_minimum_Electronegativity))
((|ElementProperty_MagpieData_mean_MeltingT - ElementProperty_MagpieData_avg_dev_MeltingT|) * (XRDPowderPattern_xrd_37 * CrystalNNFingerprint_mean_linear_CN_2))
((VoronoiFingerprint_std_dev_Voro_vol_sum / CrystalNNFingerprint_mean_hexagonal_planar_CN_6) / (|DensityFeatures_packing_fraction - ElementProperty_MagpieData_minimum_Electronegativity|))
((CrystalNNFingerprint_mean_wt_CN_6⁶) / (ValenceOrbital_frac_f_valence_electrons - BondOrientationParameter_mean_BOOP_Q_l=4))
((ElementProperty_MagpieData_avg_dev_Electronegativity * ElementProperty_MagpieData_maximum_NUnfilled) * (exp(-1.0 * ElementProperty_MagpieData_maximum_GSbandgap)))
((|ElementProperty_MagpieData_minimum_Number - ElementProperty_MagpieData_maximum_NdValence|) * (ℓ-OFM_v1_72 + ElementProperty_MagpieData_maximum_NdUnfilled))
((ElementProperty_MagpieData_avg_dev_Electronegativity²) * (exp(-1.0 * ElementProperty_MagpieData_maximum_GSbandgap)))
((CrystalNNFingerprint_mean_square_pyramidal_CN_5 / BondOrientationParameter_mean_BOOP_Q_l=1) / (BondOrientationParameter_mean_BOOP_Q_l=1 * ElementProperty_MagpieData_minimum_MendeleevNumber))
((ElementProperty_MagpieData_avg_dev_NpValence * ElementProperty_MagpieData_maximum_NdValence) - (ElementProperty_MagpieData_maximum_NdValence + ElementProperty_MagpieData_maximum_NdUnfilled))
((LocalPropertyDifference_std_dev_local_difference_in_Electronegativity * ElementProperty_MagpieData_avg_dev_Electronegativity) * (exp(-1.0 * ElementProperty_MagpieData_maximum_GSbandgap)))
((ElementProperty_MagpieData_avg_dev_NpValence / ElementProperty_MagpieData_minimum_MendeleevNumber) + (GaussianSymmFunc_std_dev_G2_80_0 / AGNIFingerPrint_std_dev_AGNI_eta=4_43e+00))
((CrystalNNFingerprint_mean_pentagonal_planar_CN_5 / BondOrientationParameter_mean_BOOP_Q_l=1) / (BondOrientationParameter_mean_BOOP_Q_l=1 * ElementProperty_MagpieData_minimum_MendeleevNumber))
((ElementProperty_MagpieData_avg_dev_Electronegativity - ElementProperty_MagpieData_avg_dev_GSbandgap) / (|ℓ-OFM_v1_172 - CrystalNNFingerprint_std_dev_octahedral_CN_6|))
((ElementProperty_MagpieData_maximum_MeltingT * GaussianSymmFunc_std_dev_G2_80_0) - (ElementProperty_MagpieData_minimum_Electronegativity * ElementProperty_MagpieData_maximum_NUnfilled))
((AGNIFingerPrint_std_dev_AGNI_dir=x_eta=1_88e+00 - BondFractions_O_-_Sn_bond_frac_) + (DensityFeatures_packing_fraction * CrystalNNFingerprint_mean_wt_CN_2))
((ElementProperty_MagpieData_maximum_MeltingT * GaussianSymmFunc_std_dev_G2_80_0) - (ℓ-OFM_v1_72 + ElementProperty_MagpieData_maximum_NUnfilled))
((AGNIFingerPrint_std_dev_AGNI_eta=4_43e+00 * ElementProperty_MagpieData_minimum_MendeleevNumber) * (ElementProperty_MagpieData_minimum_Electronegativity - ElementProperty_MagpieData_mean_NUnfilled))
((DensityFeatures_packing_fraction * CrystalNNFingerprint_mean_wt_CN_2) + sqrt(BondOrientationParameter_mean_BOOP_Q_l=1))
((ElementProperty_MagpieData_maximum_MeltingT * GaussianSymmFunc_std_dev_G2_80_0) - (ℓ-OFM_v1_72 + ElementProperty_MagpieData_mean_NUnfilled))
((CrystalNNFingerprint_mean_wt_CN_5 * SineCoulombMatrix_sine_coulomb_matrix_eig_1) * (DensityFeatures_packing_fraction * ElementProperty_MagpieData_maximum_NdUnfilled))
((ElementProperty_MagpieData_maximum_NdValence * DensityFeatures_packing_fraction) + (|ℓ-OFM_v1_72 - ElementProperty_MagpieData_maximum_NUnfilled|))
((ElementProperty_MagpieData_mean_SpaceGroupNumber / VoronoiFingerprint_mean_Voro_area_sum) / (ElementProperty_MagpieData_mean_NpValence + ElementProperty_MagpieData_maximum_NdUnfilled))
((ElementProperty_MagpieData_mean_SpaceGroupNumber / ElementProperty_MagpieData_mean_NpValence) / (ElementProperty_MagpieData_mean_NpValence + ElementProperty_MagpieData_maximum_NdUnfilled))
((ℓ-OFM_v1_164 + ElectronegativityDiff_range_EN_difference) / (|VoronoiFingerprint_mean_Voro_area_sum - CoulombMatrix_coulomb_matrix_eig_2|))
((ElementProperty_MagpieData_minimum_NsValence⁶) / (CoulombMatrix_coulomb_matrix_eig_2 * AGNIFingerPrint_std_dev_AGNI_eta=8_00e-01))
((CrystalNNFingerprint_std_dev_hexagonal_planar_CN_6 * BondFractions_Ag_-_O_bond_frac_) + (ℓ-OFM_v1_177 * GaussianSymmFunc_std_dev_G2_80_0))
((ElementProperty_MagpieData_maximum_GSvolume_pa / CrystalNNFingerprint_mean_pentagonal_pyramidal_CN_6) / (|ElectronegativityDiff_range_EN_difference - ElementProperty_MagpieData_minimum_MendeleevNumber|))
((|ElementProperty_MagpieData_mean_SpaceGroupNumber - VoronoiFingerprint_mean_Voro_area_sum|) / (ElementProperty_MagpieData_mean_NpValence + ElementProperty_MagpieData_maximum_NdUnfilled))
((CrystalNNFingerprint_std_dev_wt_CN_2 * ElectronegativityDiff_range_EN_difference) / (ElementProperty_MagpieData_mean_Column - ElementProperty_MagpieData_range_GSbandgap))
((ElementProperty_MagpieData_mean_Column * CoulombMatrix_coulomb_matrix_eig_2) + (ElementProperty_MagpieData_mean_NpValence * ElementProperty_MagpieData_mean_MeltingT))
((VoronoiFingerprint_mean_Voro_dist_minimum - ElementProperty_MagpieData_mean_Electronegativity) * (ElectronegativityDiff_minimum_EN_difference * ElectronegativityDiff_range_EN_difference))
((StructuralHeterogeneity_maximum_neighbor_distance_variation - YangSolidSolution_Yang_delta) / (|ElementProperty_MagpieData_mean_NValence - ElementProperty_MagpieData_range_GSbandgap|))
((ElectronegativityDiff_minimum_EN_difference * ElectronegativityDiff_range_EN_difference) * (CrystalNNFingerprint_mean_linear_CN_2 - ElementProperty_MagpieData_minimum_Electronegativity))
((BandCenter_band_center * CoulombMatrix_coulomb_matrix_eig_2) + (VoronoiFingerprint_mean_Voro_dist_minimum⁶))
((VoronoiFingerprint_mean_Voro_dist_minimum⁶) / (|VoronoiFingerprint_mean_Voro_dist_minimum - ElementProperty_MagpieData_minimum_MendeleevNumber|))
((ElectronegativityDiff_minimum_EN_difference * ElectronegativityDiff_range_EN_difference) * (ElementProperty_MagpieData_minimum_Electronegativity - CrystalNNFingerprint_std_dev_wt_CN_2))'''
# Clustered top features  
formulas = [line.strip() for line in text_clustered.splitlines() if line.strip()]  
  
converted_formulas_dict = {}  
for i, f in enumerate(formulas):  
    converted_dict = convert_formula(f, 'SISSO_top_cluster_feature_coGNperovsk', i + 1)  
    converted_formulas_dict.update(converted_dict)  
  
# Save the dictionary to a JSON file  
output_filename = 'sisso_clustered_topfeatures_coGNperovsk.json'  
with open(output_filename, 'w') as json_file:  
    json.dump(converted_formulas_dict, json_file, indent=4)  
  
print(f"Converted formulas saved to {output_filename}")