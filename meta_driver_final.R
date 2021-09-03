library(DescTools) #used for winsorizing data
library(caret) #used for setting up 5-fold cv
library(glmnet) #used from running elastic net
library(stats)
library(tidyverse)
library(tidyselect)
library(readxl)
library(nlme)
library(lme4)
library(MuMIn)
source("/home/max/Documents/hilo/driver_v2.R") #function that organizes model building and testing
source("/home/max/Documents/hilo/fit_enet.R") #function that build the elastic net model
source("/home/max/Documents/hilo/fit_test.R") #function that tests the elastic net model
source('/home/max/Documents/hilo/createDataFrame_maxmod.R') #function that compiles the dataset

#############################create variable name lists##################################
neo_raw <- c('NEORAW_01', 'NEORAW_02', 'NEORAW_03', 'NEORAW_04', 'NEORAW_05',
             'NEORAW_06', 'NEORAW_07', 'NEORAW_08', 'NEORAW_09', 'NEORAW_10', 
             'NEORAW_11', 'NEORAW_12', 'NEORAW_13', 'NEORAW_14', 'NEORAW_15', 
             'NEORAW_16', 'NEORAW_17', 'NEORAW_18', 'NEORAW_19', 'NEORAW_20', 
             'NEORAW_21', 'NEORAW_22', 'NEORAW_23', 'NEORAW_24', 'NEORAW_25', 
             'NEORAW_26', 'NEORAW_27', 'NEORAW_28', 'NEORAW_29', 'NEORAW_30', 
             'NEORAW_31', 'NEORAW_32', 'NEORAW_33', 'NEORAW_34', 'NEORAW_35', 
             'NEORAW_36', 'NEORAW_37', 'NEORAW_38', 'NEORAW_39', 'NEORAW_40',
             'NEORAW_41', 'NEORAW_42', 'NEORAW_43', 'NEORAW_44', 'NEORAW_45', 
             'NEORAW_46', 'NEORAW_47', 'NEORAW_48', 'NEORAW_49', 'NEORAW_50', 
             'NEORAW_51', 'NEORAW_52', 'NEORAW_53', 'NEORAW_54', 'NEORAW_55',
             'NEORAW_56', 'NEORAW_57', 'NEORAW_58', 'NEORAW_59', 'NEORAW_60')

# NEO-FFI Metatraits
metatraits <- c('neoStability', 'neoPlasticity')

# NEO-FFI Traits
traits <- c('neoN', 'neoE', 'neoO', 'neoA', 'neoC')

# NEO-FFI Facets
facets <- c('neoAnxiety', 'neoDepression', 'neoSelfConsciousness',
            'neoVulnerability', 'neoGregariousness', 'neoActivityLevel', 
            'neoCheerfulness', 'neoArtisticInterests', 'neoAdventurousness', 
            'neoIntellect', 'neoLiberalism', 'neoTrust', 'neoAltruism', 
            'neoCooperation', 'neoOrderliness', 'neoDutifulness', 
            'neoAchievementStriving', 'neoSelfDiscipline')

# Personality Variables
personality <- c(metatraits, traits, facets, neo_raw)

personality.list <- list('Meta-Trait' = metatraits, 
                         'Trait' = traits, 
                         'Facet' = facets, 
                         'Item' = neo_raw)

# Brain Variables ####
# Omnibus
omnibus <- c('FS_BrainSeg_Vol', 'FS_Total_GM_Vol', 'FS_Tot_WM_Vol', 
             'FS_TotCort_GM_Vol', 'FS_SubCort_GM_Vol', 'WhiteSurfArea_area', 
             'MeanThickness_thickness')

# Subcortical
subcortical <- c('FS_L_AccumbensArea_Vol', 'FS_L_Amygdala_Vol', 
                 'FS_L_Caudate_Vol', 'FS_L_Hippo_Vol', 'FS_L_Pallidum_Vol', 
                 'FS_L_Putamen_Vol', 'FS_L_ThalamusProper_Vol', 
                 'FS_L_VentDC_Vol', 'FS_R_AccumbensArea_Vol', 
                 'FS_R_Amygdala_Vol', 'FS_R_Caudate_Vol', 'FS_R_Hippo_Vol',
                 'FS_R_Pallidum_Vol', 'FS_R_Putamen_Vol', 
                 'FS_R_ThalamusProper_Vol', 'FS_R_VentDC_Vol',
                 'FS_L_Cerebellum_Cort_Vol', 'FS_R_Cerebellum_Cort_Vol')

# Desikan Cortical Thickness
desikanCT<- c('FS_L_Bankssts_Thck', 'FS_L_Caudalanteriorcingulate_Thck',
              'FS_L_Caudalmiddlefrontal_Thck', 'FS_L_Cuneus_Thck',
              'FS_L_Entorhinal_Thck', 'FS_L_Frontalpole_Thck', 
              'FS_L_Fusiform_Thck', 'FS_L_Inferiorparietal_Thck', 
              'FS_L_Inferiortemporal_Thck', 'FS_L_Insula_Thck',
              'FS_L_Isthmuscingulate_Thck', 'FS_L_Lateraloccipital_Thck', 
              'FS_L_Lateralorbitofrontal_Thck', 'FS_L_Lingual_Thck',
              'FS_L_Medialorbitofrontal_Thck', 'FS_L_Middletemporal_Thck', 
              'FS_L_Paracentral_Thck', 'FS_L_Parahippocampal_Thck', 
              'FS_L_Parsopercularis_Thck', 'FS_L_Parsorbitalis_Thck',
              'FS_L_Parstriangularis_Thck', 'FS_L_Pericalcarine_Thck',
              'FS_L_Postcentral_Thck', 'FS_L_Posteriorcingulate_Thck',
              'FS_L_Precentral_Thck', 'FS_L_Precuneus_Thck',
              'FS_L_Rostralanteriorcingulate_Thck', 
              'FS_L_Rostralmiddlefrontal_Thck', 'FS_L_Superiorfrontal_Thck', 
              'FS_L_Superiorparietal_Thck', 'FS_L_Superiortemporal_Thck', 
              'FS_L_Supramarginal_Thck', 'FS_L_Temporalpole_Thck', 
              'FS_L_Transversetemporal_Thck', 'FS_R_Bankssts_Thck', 
              'FS_R_Caudalanteriorcingulate_Thck', 
              'FS_R_Caudalmiddlefrontal_Thck', 'FS_R_Cuneus_Thck',
              'FS_R_Entorhinal_Thck', 'FS_R_Frontalpole_Thck',
              'FS_R_Fusiform_Thck', 'FS_R_Inferiorparietal_Thck', 
              'FS_R_Inferiortemporal_Thck', 'FS_R_Insula_Thck',
              'FS_R_Isthmuscingulate_Thck', 'FS_R_Lateraloccipital_Thck', 
              'FS_R_Lateralorbitofrontal_Thck', 'FS_R_Lingual_Thck',
              'FS_R_Medialorbitofrontal_Thck', 'FS_R_Middletemporal_Thck', 
              'FS_R_Paracentral_Thck', 'FS_R_Parahippocampal_Thck', 
              'FS_R_Parsopercularis_Thck', 'FS_R_Parsorbitalis_Thck',
              'FS_R_Parstriangularis_Thck', 'FS_R_Pericalcarine_Thck',
              'FS_R_Postcentral_Thck', 'FS_R_Posteriorcingulate_Thck',
              'FS_R_Precentral_Thck', 'FS_R_Precuneus_Thck', 
              'FS_R_Rostralanteriorcingulate_Thck', 
              'FS_R_Rostralmiddlefrontal_Thck', 'FS_R_Superiorfrontal_Thck',
              'FS_R_Superiorparietal_Thck', 'FS_R_Superiortemporal_Thck', 
              'FS_R_Supramarginal_Thck', 'FS_R_Temporalpole_Thck', 
              'FS_R_Transversetemporal_Thck')

# Desikan Cortial Area
desikanCA <- c('FS_L_Bankssts_Area', 'FS_L_Caudalanteriorcingulate_Area',
               'FS_L_Caudalmiddlefrontal_Area', 'FS_L_Cuneus_Area', 
               'FS_L_Entorhinal_Area', 'FS_L_Frontalpole_Area',
               'FS_L_Fusiform_Area', 'FS_L_Inferiorparietal_Area',
               'FS_L_Inferiortemporal_Area', 'FS_L_Insula_Area',
               'FS_L_Isthmuscingulate_Area', 'FS_L_Lateraloccipital_Area',
               'FS_L_Lateralorbitofrontal_Area', 'FS_L_Lingual_Area',
               'FS_L_Medialorbitofrontal_Area', 'FS_L_Middletemporal_Area',
               'FS_L_Paracentral_Area', 'FS_L_Parahippocampal_Area', 
               'FS_L_Parsopercularis_Area', 'FS_L_Parsorbitalis_Area',
               'FS_L_Parstriangularis_Area', 'FS_L_Pericalcarine_Area',
               'FS_L_Postcentral_Area', 'FS_L_Posteriorcingulate_Area',
               'FS_L_Precentral_Area', 'FS_L_Precuneus_Area',
               'FS_L_Rostralanteriorcingulate_Area', 
               'FS_L_Rostralmiddlefrontal_Area', 'FS_L_Superiorfrontal_Area',
               'FS_L_Superiorparietal_Area', 'FS_L_Superiortemporal_Area', 
               'FS_L_Supramarginal_Area', 'FS_L_Temporalpole_Area',
               'FS_L_Transversetemporal_Area', 'FS_R_Bankssts_Area',
               'FS_R_Caudalanteriorcingulate_Area',
               'FS_R_Caudalmiddlefrontal_Area', 'FS_R_Cuneus_Area',
               'FS_R_Entorhinal_Area', 'FS_R_Frontalpole_Area',
               'FS_R_Fusiform_Area', 'FS_R_Inferiorparietal_Area', 
               'FS_R_Inferiortemporal_Area', 'FS_R_Insula_Area',
               'FS_R_Isthmuscingulate_Area', 'FS_R_Lateraloccipital_Area', 
               'FS_R_Lateralorbitofrontal_Area', 'FS_R_Lingual_Area',
               'FS_R_Medialorbitofrontal_Area', 'FS_R_Middletemporal_Area',
               'FS_R_Paracentral_Area', 'FS_R_Parahippocampal_Area', 
               'FS_R_Parsopercularis_Area', 'FS_R_Parsorbitalis_Area', 
               'FS_R_Parstriangularis_Area', 'FS_R_Pericalcarine_Area', 
               'FS_R_Postcentral_Area', 'FS_R_Posteriorcingulate_Area', 
               'FS_R_Precentral_Area', 'FS_R_Precuneus_Area',
               'FS_R_Rostralanteriorcingulate_Area', 
               'FS_R_Rostralmiddlefrontal_Area', 'FS_R_Superiorfrontal_Area',
               'FS_R_Superiorparietal_Area', 'FS_R_Superiortemporal_Area',
               'FS_R_Supramarginal_Area', 'FS_R_Temporalpole_Area',
               'FS_R_Transversetemporal_Area')

# Destrieux Cortical Thickness
destrieuxCT <- c('rh_G_and_S_frontomargin_thickness',
                 'rh_G_and_S_occipital_inf_thickness',
                 'rh_G_and_S_paracentral_thickness',
                 'rh_G_and_S_subcentral_thickness',
                 'rh_G_and_S_transv_frontopol_thickness',
                 'rh_G_and_S_cingul-Ant_thickness',
                 'rh_G_and_S_cingul-Mid-Ant_thickness',
                 'rh_G_and_S_cingul-Mid-Post_thickness', 
                 'rh_G_cingul-Post-dorsal_thickness',
                 'rh_G_cingul-Post-ventral_thickness', 'rh_G_cuneus_thickness', 
                 'rh_G_front_inf-Opercular_thickness',
                 'rh_G_front_inf-Orbital_thickness', 
                 'rh_G_front_inf-Triangul_thickness',
                 'rh_G_front_middle_thickness', 'rh_G_front_sup_thickness',
                 'rh_G_Ins_lg_and_S_cent_ins_thickness',
                 'rh_G_insular_short_thickness', 
                 'rh_G_occipital_middle_thickness',
                 'rh_G_occipital_sup_thickness',
                 'rh_G_oc-temp_lat-fusifor_thickness',
                 'rh_G_oc-temp_med-Lingual_thickness',
                 'rh_G_oc-temp_med-Parahip_thickness', 'rh_G_orbital_thickness',
                 'rh_G_pariet_inf-Angular_thickness',
                 'rh_G_pariet_inf-Supramar_thickness',
                 'rh_G_parietal_sup_thickness', 'rh_G_postcentral_thickness',
                 'rh_G_precentral_thickness', 'rh_G_precuneus_thickness',
                 'rh_G_rectus_thickness', 'rh_G_subcallosal_thickness',
                 'rh_G_temp_sup-G_T_transv_thickness',
                 'rh_G_temp_sup-Lateral_thickness',
                 'rh_G_temp_sup-Plan_polar_thickness',
                 'rh_G_temp_sup-Plan_tempo_thickness', 
                 'rh_G_temporal_inf_thickness', 
                 'rh_G_temporal_middle_thickness',
                 'rh_Lat_Fis-ant-Horizont_thickness',
                 'rh_Lat_Fis-ant-Vertical_thickness', 
                 'rh_Lat_Fis-post_thickness', 'rh_Pole_occipital_thickness',
                 'rh_Pole_temporal_thickness', 'rh_S_calcarine_thickness',
                 'rh_S_central_thickness', 'rh_S_cingul-Marginalis_thickness',
                 'rh_S_circular_insula_ant_thickness',
                 'rh_S_circular_insula_inf_thickness',	
                 'rh_S_circular_insula_sup_thickness', 
                 'rh_S_collat_transv_ant_thickness',
                 'rh_S_collat_transv_post_thickness', 
                 'rh_S_front_inf_thickness', 'rh_S_front_middle_thickness',
                 'rh_S_front_sup_thickness', 
                 'rh_S_interm_prim-Jensen_thickness',
                 'rh_S_intrapariet_and_P_trans_thickness',
                 'rh_S_oc_middle_and_Lunatus_thickness',
                 'rh_S_oc_sup_and_transversal_thickness',
                 'rh_S_occipital_ant_thickness',
                 'rh_S_oc-temp_lat_thickness',
                 'rh_S_oc-temp_med_and_Lingual_thickness',
                 'rh_S_orbital_lateral_thickness',
                 'rh_S_orbital_med-olfact_thickness',
                 'rh_S_orbital-H_Shaped_thickness',
                 'rh_S_parieto_occipital_thickness',	
                 'rh_S_pericallosal_thickness',
                 'rh_S_postcentral_thickness',
                 'rh_S_precentral-inf-part_thickness',
                 'rh_S_precentral-sup-part_thickness',
                 'rh_S_suborbital_thickness', 'rh_S_subparietal_thickness',
                 'rh_S_temporal_inf_thickness', 'rh_S_temporal_sup_thickness',
                 'rh_S_temporal_transverse_thickness', 
                 'lh_G_and_S_frontomargin_thickness', 
                 'lh_G_and_S_occipital_inf_thickness',
                 'lh_G_and_S_paracentral_thickness',
                 'lh_G_and_S_subcentral_thickness',
                 'lh_G_and_S_transv_frontopol_thickness',
                 'lh_G_and_S_cingul-Ant_thickness',
                 'lh_G_and_S_cingul-Mid-Ant_thickness',
                 'lh_G_and_S_cingul-Mid-Post_thickness',
                 'lh_G_cingul-Post-dorsal_thickness',
                 'lh_G_cingul-Post-ventral_thickness', 'lh_G_cuneus_thickness', 
                 'lh_G_front_inf-Opercular_thickness',
                 'lh_G_front_inf-Orbital_thickness',
                 'lh_G_front_inf-Triangul_thickness',
                 'lh_G_front_middle_thickness',	'lh_G_front_sup_thickness',
                 'lh_G_Ins_lg_and_S_cent_ins_thickness',
                 'lh_G_insular_short_thickness',
                 'lh_G_occipital_middle_thickness',
                 'lh_G_occipital_sup_thickness',
                 'lh_G_oc-temp_lat-fusifor_thickness',
                 'lh_G_oc-temp_med-Lingual_thickness',
                 'lh_G_oc-temp_med-Parahip_thickness', 'lh_G_orbital_thickness',
                 'lh_G_pariet_inf-Angular_thickness',
                 'lh_G_pariet_inf-Supramar_thickness',
                 'lh_G_parietal_sup_thickness', 'lh_G_postcentral_thickness',
                 'lh_G_precentral_thickness', 'lh_G_precuneus_thickness',
                 'lh_G_rectus_thickness', 'lh_G_subcallosal_thickness',
                 'lh_G_temp_sup-G_T_transv_thickness',
                 'lh_G_temp_sup-Lateral_thickness',
                 'lh_G_temp_sup-Plan_polar_thickness',
                 'lh_G_temp_sup-Plan_tempo_thickness',
                 'lh_G_temporal_inf_thickness',
                 'lh_G_temporal_middle_thickness',
                 'lh_Lat_Fis-ant-Horizont_thickness',
                 'lh_Lat_Fis-ant-Vertical_thickness', 
                 'lh_Lat_Fis-post_thickness', 'lh_Pole_occipital_thickness',
                 'lh_Pole_temporal_thickness', 'lh_S_calcarine_thickness',
                 'lh_S_central_thickness', 'lh_S_cingul-Marginalis_thickness',
                 'lh_S_circular_insula_ant_thickness',
                 'lh_S_circular_insula_inf_thickness',	
                 'lh_S_circular_insula_sup_thickness',
                 'lh_S_collat_transv_ant_thickness',
                 'lh_S_collat_transv_post_thickness', 
                 'lh_S_front_inf_thickness', 'lh_S_front_middle_thickness',
                 'lh_S_front_sup_thickness', 
                 'lh_S_interm_prim-Jensen_thickness',
                 'lh_S_intrapariet_and_P_trans_thickness',
                 'lh_S_oc_middle_and_Lunatus_thickness',
                 'lh_S_oc_sup_and_transversal_thickness',
                 'lh_S_occipital_ant_thickness', 'lh_S_oc-temp_lat_thickness',
                 'lh_S_oc-temp_med_and_Lingual_thickness',
                 'lh_S_orbital_lateral_thickness',
                 'lh_S_orbital_med-olfact_thickness',
                 'lh_S_orbital-H_Shaped_thickness',
                 'lh_S_parieto_occipital_thickness',
                 'lh_S_pericallosal_thickness', 'lh_S_postcentral_thickness',
                 'lh_S_precentral-inf-part_thickness',
                 'lh_S_precentral-sup-part_thickness',
                 'lh_S_suborbital_thickness', 'lh_S_subparietal_thickness',
                 'lh_S_temporal_inf_thickness', 'lh_S_temporal_sup_thickness',
                 'lh_S_temporal_transverse_thickness')

# Destrieux Cortical Area
destrieuxCA <- c('rh_G_and_S_frontomargin_area', 
                 'rh_G_and_S_occipital_inf_area', 'rh_G_and_S_paracentral_area',	
                 'rh_G_and_S_subcentral_area',
                 'rh_G_and_S_transv_frontopol_area',
                 'rh_G_and_S_cingul-Ant_area', 'rh_G_and_S_cingul-Mid-Ant_area',
                 'rh_G_and_S_cingul-Mid-Post_area',	
                 'rh_G_cingul-Post-dorsal_area', 
                 'rh_G_cingul-Post-ventral_area',	'rh_G_cuneus_area',	
                 'rh_G_front_inf-Opercular_area',	'rh_G_front_inf-Orbital_area',	
                 'rh_G_front_inf-Triangul_area', 'rh_G_front_middle_area',
                 'rh_G_front_sup_area', 'rh_G_Ins_lg_and_S_cent_ins_area',
                 'rh_G_insular_short_area',	'rh_G_occipital_middle_area',
                 'rh_G_occipital_sup_area',	'rh_G_oc-temp_lat-fusifor_area',
                 'rh_G_oc-temp_med-Lingual_area',	
                 'rh_G_oc-temp_med-Parahip_area',	'rh_G_orbital_area',
                 'rh_G_pariet_inf-Angular_area',	
                 'rh_G_pariet_inf-Supramar_area',	'rh_G_parietal_sup_area',
                 'rh_G_postcentral_area',	'rh_G_precentral_area',
                 'rh_G_precuneus_area', 'rh_G_rectus_area', 
                 'rh_G_subcallosal_area',	'rh_G_temp_sup-G_T_transv_area',
                 'rh_G_temp_sup-Lateral_area', 'rh_G_temp_sup-Plan_polar_area',	
                 'rh_G_temp_sup-Plan_tempo_area',	'rh_G_temporal_inf_area',
                 'rh_G_temporal_middle_area',	'rh_Lat_Fis-ant-Horizont_area',	
                 'rh_Lat_Fis-ant-Vertical_area', 'rh_Lat_Fis-post_area',
                 'rh_Pole_occipital_area', 'rh_Pole_temporal_area',
                 'rh_S_calcarine_area',	'rh_S_central_area', 
                 'rh_S_cingul-Marginalis_area',	'rh_S_circular_insula_ant_area',	
                 'rh_S_circular_insula_inf_area',	
                 'rh_S_circular_insula_sup_area',	'rh_S_collat_transv_ant_area',	
                 'rh_S_collat_transv_post_area', 'rh_S_front_inf_area',	
                 'rh_S_front_middle_area', 'rh_S_front_sup_area',
                 'rh_S_interm_prim-Jensen_area',
                 'rh_S_intrapariet_and_P_trans_area',
                 'rh_S_oc_middle_and_Lunatus_area', 
                 'rh_S_oc_sup_and_transversal_area', 'rh_S_occipital_ant_area',
                 'rh_S_oc-temp_lat_area', 'rh_S_oc-temp_med_and_Lingual_area',	
                 'rh_S_orbital_lateral_area',	'rh_S_orbital_med-olfact_area',
                 'rh_S_orbital-H_Shaped_area', 'rh_S_parieto_occipital_area',	
                 'rh_S_pericallosal_area', 'rh_S_postcentral_area',	
                 'rh_S_precentral-inf-part_area',	
                 'rh_S_precentral-sup-part_area',	'rh_S_suborbital_area',
                 'rh_S_subparietal_area',	'rh_S_temporal_inf_area',	
                 'rh_S_temporal_sup_area', 'rh_S_temporal_transverse_area',
                 'lh_G_and_S_frontomargin_area',
                 'lh_G_and_S_occipital_inf_area', 'lh_G_and_S_paracentral_area',	
                 'lh_G_and_S_subcentral_area',
                 'lh_G_and_S_transv_frontopol_area',
                 'lh_G_and_S_cingul-Ant_area', 'lh_G_and_S_cingul-Mid-Ant_area',
                 'lh_G_and_S_cingul-Mid-Post_area',
                 'lh_G_cingul-Post-dorsal_area',
                 'lh_G_cingul-Post-ventral_area',	'lh_G_cuneus_area',
                 'lh_G_front_inf-Opercular_area',	'lh_G_front_inf-Orbital_area',
                 'lh_G_front_inf-Triangul_area', 'lh_G_front_middle_area',
                 'lh_G_front_sup_area',	'lh_G_Ins_lg_and_S_cent_ins_area',
                 'lh_G_insular_short_area',	'lh_G_occipital_middle_area',
                 'lh_G_occipital_sup_area',	'lh_G_oc-temp_lat-fusifor_area',
                 'lh_G_oc-temp_med-Lingual_area',
                 'lh_G_oc-temp_med-Parahip_area',	'lh_G_orbital_area',
                 'lh_G_pariet_inf-Angular_area', 
                 'lh_G_pariet_inf-Supramar_area', 'lh_G_parietal_sup_area',	
                 'lh_G_postcentral_area', 'lh_G_precentral_area',
                 'lh_G_precuneus_area',	'lh_G_rectus_area', 
                 'lh_G_subcallosal_area',	'lh_G_temp_sup-G_T_transv_area',
                 'lh_G_temp_sup-Lateral_area', 'lh_G_temp_sup-Plan_polar_area',
                 'lh_G_temp_sup-Plan_tempo_area',	'lh_G_temporal_inf_area',
                 'lh_G_temporal_middle_area',	'lh_Lat_Fis-ant-Horizont_area',
                 'lh_Lat_Fis-ant-Vertical_area',	'lh_Lat_Fis-post_area',
                 'lh_Pole_occipital_area',	'lh_Pole_temporal_area',
                 'lh_S_calcarine_area',	'lh_S_central_area',
                 'lh_S_cingul-Marginalis_area', 'lh_S_circular_insula_ant_area',
                 'lh_S_circular_insula_inf_area',
                 'lh_S_circular_insula_sup_area', 'lh_S_collat_transv_ant_area',
                 'lh_S_collat_transv_post_area', 'lh_S_front_inf_area',
                 'lh_S_front_middle_area', 'lh_S_front_sup_area',
                 'lh_S_interm_prim-Jensen_area', 
                 'lh_S_intrapariet_and_P_trans_area',
                 'lh_S_oc_middle_and_Lunatus_area',
                 'lh_S_oc_sup_and_transversal_area',	
                 'lh_S_occipital_ant_area',	'lh_S_oc-temp_lat_area',
                 'lh_S_oc-temp_med_and_Lingual_area',
                 'lh_S_orbital_lateral_area',	'lh_S_orbital_med-olfact_area',
                 'lh_S_orbital-H_Shaped_area', 'lh_S_parieto_occipital_area', 
                 'lh_S_pericallosal_area', 'lh_S_postcentral_area', 
                 'lh_S_precentral-inf-part_area', 
                 'lh_S_precentral-sup-part_area',	'lh_S_suborbital_area',
                 'lh_S_subparietal_area',	'lh_S_temporal_inf_area',
                 'lh_S_temporal_sup_area',	'lh_S_temporal_transverse_area')

# Brain Variables
brain <- c(all_of(omnibus), all_of(subcortical), all_of(desikanCT), 
           all_of(desikanCA), all_of(destrieuxCT), all_of(destrieuxCA))

brain.region <- list(omnibus = omnibus, 
                     subcortical = subcortical, 
                     desikanCT = desikanCT, 
                     desikanCA = desikanCA, 
                     desikan = c(desikanCT, desikanCA),
                     destrieuxCT = destrieuxCT,
                     destrieuxCA = destrieuxCA,
                     destrieux = c(destrieuxCT, destrieuxCA))

desikan_vars <- c(brain.region$desikan,brain.region$subcortical,brain.region$omnibus)
destrieux_vars <- c(brain.region$destrieux,brain.region$subcortical,brain.region$omnibus)
all_personality_vars <- c(personality.list[[1]],personality.list[[2]],personality.list[[3]],personality.list[[4]])

##########MAKE TRAIN/TEST LISTS########################
Family_id_unique <- unique(data$Family_ID)
test_percent <- .2
test_size <- round(length(Family_id_unique)*test_percent)
test_famids <- list()
for (iter in 1:10) {
  test_famids[[iter]] <- sample(Family_id_unique, size=test_size, replace = FALSE, prob = NULL)
}
#read in IDs after you've assigned so they stay the same throughout analyses
#test_famids =  readRDS( paste0("/home/max/Documents/hilo/test_famids.rds"))

#############EN ANALYSES#######################
#function that runs the elastic nets
en_loop=function(y){
  output_list_N_desikan <- lapply(1:10, driver_v2, y, desikan_vars, data, test_famids)
  output_list_N_destrieux <- lapply(1:10, driver_v2, y, destrieux_vars, data, test_famids)
  return(list(output_list_N_desikan,output_list_N_destrieux))
}

#create output holder
output_list <- list()

#run en analyses
output_list <- lapply(all_personality_vars, en_loop)

#load in analyses after finishing so you don't have to run again
#output_list =  readRDS( paste0("/home/max/Documents/hilo/output_of_hilo_en.rds"))

###########compile EN results################
#make desikan table
cv_desikan <- data.frame(best_cv_desikan = 1:length(all_personality_vars), mean_cv_desikan = 1:length(all_personality_vars), mean_test_desikan = 1:length(all_personality_vars), mean_test_new_desikan = 1:length(all_personality_vars), row.names = all_personality_vars)

for (ii in 1:length(all_personality_vars)){
cv_desikan[ii, 1] <- rowMeans(data.frame(output_list[[ii]][[1]][[1]][[1]], output_list[[ii]][[1]][[2]][[1]], output_list[[ii]][[1]][[3]][[1]], output_list[[ii]][[1]][[4]][[1]], output_list[[ii]][[1]][[5]][[1]], output_list[[ii]][[1]][[6]][[1]], output_list[[ii]][[1]][[7]][[1]], output_list[[ii]][[1]][[8]][[1]], output_list[[ii]][[1]][[9]][[1]], output_list[[ii]][[1]][[10]][[1]]))

cv_desikan[ii, 2] <- rowMeans(data.frame(output_list[[ii]][[1]][[1]][[2]], output_list[[ii]][[1]][[2]][[2]], output_list[[ii]][[1]][[3]][[2]], output_list[[ii]][[1]][[4]][[2]], output_list[[ii]][[1]][[5]][[2]], output_list[[ii]][[1]][[6]][[2]], output_list[[ii]][[1]][[7]][[2]], output_list[[ii]][[1]][[8]][[2]], output_list[[ii]][[1]][[9]][[2]],output_list[[ii]][[1]][[10]][[2]]))

cv_desikan[ii, 3] <- rowMeans(data.frame(output_list[[ii]][[1]][[1]][[3]], output_list[[ii]][[1]][[2]][[3]], output_list[[ii]][[1]][[3]][[3]], output_list[[ii]][[1]][[4]][[3]], output_list[[ii]][[1]][[5]][[3]], output_list[[ii]][[1]][[6]][[3]], output_list[[ii]][[1]][[7]][[3]], output_list[[ii]][[1]][[8]][[3]], output_list[[ii]][[1]][[9]][[3]], output_list[[ii]][[1]][[10]][[3]]))

cv_desikan[ii, 4] <- rowMeans(data.frame(output_list[[ii]][[1]][[1]][[4]], output_list[[ii]][[1]][[2]][[4]], output_list[[ii]][[1]][[3]][[4]], output_list[[ii]][[1]][[4]][[4]], output_list[[ii]][[1]][[5]][[4]], output_list[[ii]][[1]][[6]][[4]], output_list[[ii]][[1]][[7]][[4]], output_list[[ii]][[1]][[8]][[4]], output_list[[ii]][[1]][[9]][[4]], output_list[[ii]][[1]][[10]][[4]]))
}

#make destrieux table
cv_dest <- data.frame(best_cv_desikan = 1:length(all_personality_vars), mean_cv_desikan = 1:length(all_personality_vars), mean_test_desikan = 1:length(all_personality_vars), mean_test_new_desikan = 1:length(all_personality_vars), row.names = all_personality_vars)

for (ii in 1:length(all_personality_vars)){
  cv_dest[ii, 1] <- rowMeans(data.frame(output_list[[ii]][[2]][[1]][[1]], output_list[[ii]][[2]][[2]][[1]], output_list[[ii]][[2]][[3]][[1]], output_list[[ii]][[2]][[4]][[1]], output_list[[ii]][[2]][[5]][[1]], output_list[[ii]][[2]][[6]][[1]], output_list[[ii]][[2]][[7]][[1]], output_list[[ii]][[2]][[8]][[1]], output_list[[ii]][[2]][[9]][[1]], output_list[[ii]][[2]][[10]][[1]]))
  
  cv_dest[ii, 2] <- rowMeans(data.frame(output_list[[ii]][[2]][[1]][[2]], output_list[[ii]][[2]][[2]][[2]], output_list[[ii]][[2]][[3]][[2]], output_list[[ii]][[2]][[4]][[2]], output_list[[ii]][[2]][[5]][[2]], output_list[[ii]][[2]][[6]][[2]], output_list[[ii]][[2]][[7]][[2]], output_list[[ii]][[2]][[8]][[2]], output_list[[ii]][[2]][[9]][[2]],  output_list[[ii]][[2]][[10]][[2]]))
  
  cv_dest[ii, 3] <- rowMeans(data.frame(output_list[[ii]][[2]][[1]][[3]], output_list[[ii]][[2]][[2]][[3]], output_list[[ii]][[2]][[3]][[3]], output_list[[ii]][[2]][[4]][[3]], output_list[[ii]][[2]][[5]][[3]], output_list[[ii]][[2]][[6]][[3]], output_list[[ii]][[2]][[7]][[3]], output_list[[ii]][[2]][[8]][[3]], output_list[[ii]][[2]][[9]][[3]],  output_list[[ii]][[2]][[10]][[3]]))
 
   cv_dest[ii, 4] <- rowMeans(data.frame(output_list[[ii]][[2]][[1]][[4]], output_list[[ii]][[2]][[2]][[4]], output_list[[ii]][[2]][[3]][[4]],  output_list[[ii]][[2]][[4]][[4]], output_list[[ii]][[2]][[5]][[4]], output_list[[ii]][[2]][[6]][[4]], output_list[[ii]][[2]][[7]][[4]], output_list[[ii]][[2]][[8]][[4]], output_list[[ii]][[2]][[9]][[4]], output_list[[ii]][[2]][[10]][[4]]))
}

write.csv(cv_desikan,'/home/max/Documents/hilo/Desikan_en_results_hilo_part_all.csv')
write.csv(cv_dest,'/home/max/Documents/hilo/Destrieux_en_results_hilo_part_all.csv')

##############get SDs#####################
library(matrixStats)
sd_desikan <- data.frame(best_cv_desikan = 1:length(all_personality_vars), mean_cv_desikan = 1:length(all_personality_vars), mean_test_desikan = 1:length(all_personality_vars), mean_test_new_desikan = 1:length(all_personality_vars), row.names = all_personality_vars)

for (ii in 1:length(all_personality_vars)){
  sd_desikan[ii, 1] <- rowSds(as.matrix(data.frame(output_list[[ii]][[1]][[1]][[1]], output_list[[ii]][[1]][[2]][[1]], output_list[[ii]][[1]][[3]][[1]], output_list[[ii]][[1]][[4]][[1]], output_list[[ii]][[1]][[5]][[1]], output_list[[ii]][[1]][[6]][[1]], output_list[[ii]][[1]][[7]][[1]], output_list[[ii]][[1]][[8]][[1]], output_list[[ii]][[1]][[9]][[1]], output_list[[ii]][[1]][[10]][[1]])))
  
  sd_desikan[ii, 2] <- rowSds(as.matrix(data.frame(output_list[[ii]][[1]][[1]][[2]], output_list[[ii]][[1]][[2]][[2]], output_list[[ii]][[1]][[3]][[2]], output_list[[ii]][[1]][[4]][[2]], output_list[[ii]][[1]][[5]][[2]], output_list[[ii]][[1]][[6]][[2]], output_list[[ii]][[1]][[7]][[2]], output_list[[ii]][[1]][[8]][[2]], output_list[[ii]][[1]][[9]][[2]],output_list[[ii]][[1]][[10]][[2]])))
  
  sd_desikan[ii, 3] <- rowSds(as.matrix(data.frame(output_list[[ii]][[1]][[1]][[3]], output_list[[ii]][[1]][[2]][[3]], output_list[[ii]][[1]][[3]][[3]], output_list[[ii]][[1]][[4]][[3]], output_list[[ii]][[1]][[5]][[3]], output_list[[ii]][[1]][[6]][[3]], output_list[[ii]][[1]][[7]][[3]], output_list[[ii]][[1]][[8]][[3]], output_list[[ii]][[1]][[9]][[3]], output_list[[ii]][[1]][[10]][[3]])))
  
  sd_desikan[ii, 4] <- rowSds(as.matrix(data.frame(output_list[[ii]][[1]][[1]][[4]], output_list[[ii]][[1]][[2]][[4]], output_list[[ii]][[1]][[3]][[4]], output_list[[ii]][[1]][[4]][[4]], output_list[[ii]][[1]][[5]][[4]], output_list[[ii]][[1]][[6]][[4]], output_list[[ii]][[1]][[7]][[4]], output_list[[ii]][[1]][[8]][[4]], output_list[[ii]][[1]][[9]][[4]], output_list[[ii]][[1]][[10]][[4]])))
}

#make destrieux table
sd_dest <- data.frame(best_cv_dest = 1:length(all_personality_vars), mean_cv_dest = 1:length(all_personality_vars), mean_test_dest = 1:length(all_personality_vars), mean_test_new_dest= 1:length(all_personality_vars), row.names = all_personality_vars)

for (ii in 1:length(all_personality_vars)){
  sd_dest[ii, 1] <- rowSds(as.matrix(data.frame(output_list[[ii]][[2]][[1]][[1]], output_list[[ii]][[2]][[2]][[1]], output_list[[ii]][[2]][[3]][[1]], output_list[[ii]][[2]][[4]][[1]], output_list[[ii]][[2]][[5]][[1]], output_list[[ii]][[2]][[6]][[1]], output_list[[ii]][[2]][[7]][[1]], output_list[[ii]][[2]][[8]][[1]], output_list[[ii]][[2]][[9]][[1]], output_list[[ii]][[2]][[10]][[1]])))
  
  sd_dest[ii, 2] <- rowSds(as.matrix(data.frame(output_list[[ii]][[2]][[1]][[2]], output_list[[ii]][[2]][[2]][[2]], output_list[[ii]][[2]][[3]][[2]], output_list[[ii]][[2]][[4]][[2]], output_list[[ii]][[2]][[5]][[2]], output_list[[ii]][[2]][[6]][[2]], output_list[[ii]][[2]][[7]][[2]], output_list[[ii]][[2]][[8]][[2]], output_list[[ii]][[2]][[9]][[2]],  output_list[[ii]][[2]][[10]][[2]])))
  
  sd_dest[ii, 3] <- rowSds(as.matrix(data.frame(output_list[[ii]][[2]][[1]][[3]], output_list[[ii]][[2]][[2]][[3]], output_list[[ii]][[2]][[3]][[3]], output_list[[ii]][[2]][[4]][[3]], output_list[[ii]][[2]][[5]][[3]], output_list[[ii]][[2]][[6]][[3]], output_list[[ii]][[2]][[7]][[3]], output_list[[ii]][[2]][[8]][[3]], output_list[[ii]][[2]][[9]][[3]],  output_list[[ii]][[2]][[10]][[3]])))
  
  sd_dest[ii, 4] <- rowSds(as.matrix(data.frame(output_list[[ii]][[2]][[1]][[4]], output_list[[ii]][[2]][[2]][[4]], output_list[[ii]][[2]][[3]][[4]],  output_list[[ii]][[2]][[4]][[4]], output_list[[ii]][[2]][[5]][[4]], output_list[[ii]][[2]][[6]][[4]], output_list[[ii]][[2]][[7]][[4]], output_list[[ii]][[2]][[8]][[4]], output_list[[ii]][[2]][[9]][[4]], output_list[[ii]][[2]][[10]][[4]])))
}

write.csv(sd_desikan,'/home/max/Documents/hilo/Desikan_en_sds_hilo_part_all.csv')
write.csv(sd_dest,'/home/max/Documents/hilo/Destrieux_en_sds_hilo_part_all.csv')
