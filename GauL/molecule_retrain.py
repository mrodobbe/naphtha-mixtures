from GauL.GauLsrc.representation import load_representations
from GauL.GauLsrc.makeMolecule import molecule_list, normalize, input_checker, store_bad_molecules
from GauL.GauLsrc.crossDouble import training, write_statistics
from GauL.GauLsrc.plots import output_plot
from GauL.GauLsrc.results_processing import train_results_to_logfile
import sys
import time

start = time.time()

input_checker(sys.argv, "retrain")

molecule_file = sys.argv[1]
target_property = str(sys.argv[2])
save_folder = sys.argv[3]

molecules, outputs, conformers, bad_molecules = molecule_list(molecule_file)
store_bad_molecules(bad_molecules, save_folder)
output_plot(molecules, outputs, target_property, folder=save_folder)

representations, molecules = load_representations(molecules, conformers, save_folder)
outputs, heavy_atoms = normalize(molecules, outputs, target_property)

n_folds = 10  # TODO: Make argument
cv_info = training(molecules, heavy_atoms, representations, outputs, save_folder, target_property, n_folds)

end = time.time()
time_elapsed = end-start

results_list = write_statistics(cv_info, target_property, n_folds, time_elapsed, save_folder)
train_results_to_logfile(molecules, outputs, results_list, representations,
                         target_property, molecule_file, time_elapsed, save_folder)
