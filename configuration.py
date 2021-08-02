def build_config():
    cfg = type('', (), {})()
    # cfg.trial_messages_team = 'data/TrialMessages/MissionB'
    cfg.trial_messages_team = 'data/TrialMessages/SaturnB'
    # cfg.trial_messages_team = 'data/TrialMessages/test_a'
    cfg.dac = False
    cfg.dac_file = 'data/DAC/A'
    # cfg.competency_folder = 'data/CompetencyTestMessages'
    # cfg.falcon_easy_folder = 'data/FalconEasy'
    # cfg.falcon_medium_folder = 'data/FalconMedium'
    # cfg.falcon_hard_folder = 'data/FalconHard'
    cfg.zones_team = 'building_info/ASIST_Study2_SaturnMaps_Zoned.csv'
    # cfg.zones_victims = 'building_info/Victim_Starting_Zones_MissionA.csv'
    cfg.victims_team = 'building_info/ASIST_Study2_SaturnMaps_Victims_MissionB.csv'
    cfg.rubbles = 'building_info/ASIST_Study2_SaturnMaps_Rubbles_MissionB.csv'
    cfg.victime_class = 'building_info/Victim_Classes_MissionB.csv'
    cfg.rubble_class = 'building_info/Rubble_Classes_MissionB.csv'
    # cfg.zones_falcon_medium = 'building_info/falcon_zoning_medium.csv'
    # cfg.victims_falcon_medium = 'building_info/falcon_victims_coords_medium.csv'
    # cfg.zones_falcon_hard = 'building_info/falcon_zoning_hard.csv'
    # cfg.victims_falcon_hard = 'building_info/falcon_victims_coords_hard.csv'
    cfg.results_file_players = 'results/results_players.csv'
    cfg.results_file_teams = 'results/results_teams.csv'
    return cfg

