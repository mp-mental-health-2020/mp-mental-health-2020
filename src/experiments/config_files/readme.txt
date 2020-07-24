All experiment configurations which emerge from the combination of the values in the arrays will be run. Except those configurations which are explicitly given in the exclude section.

Available options:
"classification_types": ["binary", "multi"]
"experiment_dir_paths": [main_data_directory_paths]
"experiment_dirs_selected": [arrays_of_data_directories_of_single_recordings_to_be_used]
"use_indoor": [false, true]
"feature_calculation_settings": ["minimal", "efficient", "comprehensive"]
"window_sizes": [50, 100]

Example json configuration:

{
	"classification_types": ["binary"],
	"experiment_dir_paths": ["../../data/phyphox/full recordings/"],
	"experiment_dirs_selected": [["Ana-2", "Ariane", "Julian", "Wiktoria"]],
	"use_indoor": [false, true],
	"feature_calculation_settings": ["minimal", "efficient"],
	"window_sizes": [50, 100],
	"exclude":
		[
			{
				"classification_types": ["binary"],
				"experiment_dir_paths": ["../../data/phyphox/full recordings/"],
				"experiment_dirs_selected": [["Ana-2", "Ariane", "Julian", "Wiktoria"]],
				"use_indoor": [true],
				"feature_calculation_settings": ["efficient"],
				"window_sizes": [50,100]
			},
			{
				"classification_types": ["binary"],
				"experiment_dir_paths": ["../../data/phyphox/full recordings/"],
				"experiment_dirs_selected": [["Ana-2", "Ariane", "Julian", "Wiktoria"]],
				"use_indoor": [true],
				"feature_calculation_settings": ["efficient"],
				"window_sizes": [100]
			}
		]
}

=> 5 experiments will be executed (initual situation: 1*1*1*2*2*2 = 8; exclude 1: 1*1*1*1*1*2 = 2; exclude 2: 1*1*1*1*1*1 = 1; 8-3 = 5).
