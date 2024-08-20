import os
import random
from collections import defaultdict
from torch.utils.data import Dataset
from env.load_data import load_ipps
from utils.utils import solution_reader, nums_detec
import torch
from tqdm import tqdm
class ILDataScheduler:
    """
    A class for managing and processing data for the ILDataScheduler.

    Args:
        config (object): The configuration object.
        data_dir (str): The directory path where the data is stored.

    Attributes:
        config (object): The configuration object.
        data (list): The list of data.
        data_dir (str): The directory path where the data is stored.
        dir_pros (str): The directory path for problem files.
        dir_sols (str): The directory path for solution files.
        classify_dir (defaultdict): The dictionary for classified data.

    Methods:
        load_data_path: Loads the data paths.
        classify_data: Classifies the data.
        sample: Samples the data.
        processed_dir: Returns the processed data directory path.
        save_processed_data: Saves the processed data.
        load_dataset: Loads the dataset.

    """

    def __init__(self, config, data_dir = None, device = None, reload = False):
        self.config = config
        self.data = []
        self.data_dir = data_dir
        self.dir_pros = os.path.join(data_dir, 'problem')
        self.dir_sols = os.path.join(data_dir, 'correct_solution')
        self.device = torch.device('cuda') if device is None else device
        if reload:
            self.load_data_path()

    def load_data_path(self, dir_pros=None, dir_sols=None):
        """
        Loads the data paths.

        Args:
            dir_pros (str, optional): The directory path for problem files.
            dir_sols (str, optional): The directory path for solution files.

        Raises:
            AssertionError: If only one of the directories is provided.

        """
        if dir_pros is not None:
            assert dir_sols is not None, 'Please provide both directories'
        list_sols = os.listdir(self.dir_sols) if dir_sols is None else os.listdir(dir_sols)
        list_pros = os.listdir(self.dir_pros) if dir_pros is None else os.listdir(dir_pros)

        for sol in list_sols:
            prefix = sol[:-8]
            prefix = prefix[8:-8] if prefix.endswith("_optimal") else prefix[8:]
            pro = prefix + ".ipps"
            assert pro in list_pros, 'No relative problem.'
            self.data.append((os.path.join(self.dir_pros, pro), os.path.join(self.dir_sols, sol)))
            list_pros.remove(pro)

        print('-----------------Data loaded-----------------')
        print('Total num of problems with solutions: ', len(self.data))
        print('---------------------------------------------')

    def classify_data(self, classify_mas=True):
        """
        Classifies the data.

        Args:
            classify_mas (bool, optional): Whether to classify based on machine numbers.

        Returns:
            defaultdict: The dictionary containing the classified data.

        """
        classified_dir = defaultdict(list)
        for pro, sol in self.data:
            with open(pro, 'r') as f:
                job, mas, _ = f.readline().split()
                if not classify_mas:
                    classified_dir[int(job)].append((pro, sol))
                else:
                    classified_dir[(int(job), int(mas))].append((pro, sol))
        print('---------------Data classified---------------')
        print('Total num of classes: ', len(classified_dir))
        if not classify_mas:
            for k, v in classified_dir.items():
                print('-Job num: ', k, ' Num of instances: ', len(v))
        else:
            for k, v in classified_dir.items():
                print('-Job num: ', k[0], ' Machine num: ', k[1], ' Num of instances: ', len(v))
        print('---------------------------------------------')
        self.classify_dir = classified_dir
        return classified_dir

    def sample(self, num_samples, class_wise=False):
        """
        Samples the data.

        Args:
            num_samples (int): The number of samples to be returned.
            class_wise (bool, optional): Whether to sample class-wise.

        Returns:
            list: The sampled data.

        """
        if class_wise:
            choice_class = random.choice(list(self.classify_dir.keys()))
            if num_samples > len(self.classify_dir[choice_class]):
                return self.classify_dir[choice_class]
            return random.sample(self.classify_dir[choice_class], num_samples)
        if num_samples > len(self.data):
            return self.data
        return random.sample(self.data, num_samples)

    @property
    def processed_dir(self):
        """
        Returns the processed data directory path.

        Returns:
            str: The processed data directory path.

        """
        return os.path.join(self.data_dir, 'processed')

    def save_processed_data(self, update_class=None, classify_mas: bool = False):
        """
        Saves the processed data.

        Args:
            update_class (list, optional): The list of classes to update.
            classify_mas (bool, optional): Whether to classify based on machine numbers.

        """
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        class_data = self.classify_data(classify_mas)

        for k, v in class_data.items():
            if update_class is not None and k not in update_class:
                continue
            pair_dict = {'problem': {'info': [], 'tensor': [], 'name':[]}, 'solution': []}
            for pro, sol in tqdm(v, desc='Processing class: ' + str(k)):
                with open(pro, 'r') as f:
                    pro_lines = f.read().splitlines()
                prob_tensor = load_ipps(pro_lines)

                with open(sol, 'r') as f:
                    sol_lines = f.read().splitlines()
                solution = solution_reader(sol_lines, prob_tensor[4])

                pair_dict['problem']['tensor'].append(prob_tensor)
                pair_dict['problem']['info'].append(nums_detec(pro_lines))
                pair_dict['problem']['name'] .append(pro)
                pair_dict['solution'].append(solution)

            class_id = [str(i) for i in k] if type(k) == tuple else [str(k)]
            assert type(pair_dict['problem']) == dict and type(pair_dict['solution']) == list, \
                "Data must be in the form of {'problem': {'info':[], 'tensor':[]}, 'solution': []}"
            assert len(pair_dict['problem']['tensor']) == len(pair_dict['solution']), \
                "Number of problems and solutions must be equal"

            torch.save(pair_dict, os.path.join(self.processed_dir, 'data_' + '_'.join(class_id) + '.pt'))
            print('Data saved for class: ', k)
        print('----------------Data saved----------------')

    def load_dataset(self, class_ids: list = None, shuffle=True):
        """
        Loads the dataset.

        Args:
            class_ids (list, optional): The list of class IDs to load.
            shuffle (bool, optional): Whether to shuffle the dataset.

        Yields:
            ProSolDataset: The dataset object.

        """
        files = []

        if class_ids is None:
            files = [os.path.join(self.processed_dir, file) \
                     for file in os.listdir(self.processed_dir)]
        else:
            files = [os.path.join(self.processed_dir, 'data_' + '_'.join(class_id) + '.pt') \
                     for class_id in class_ids]

        if shuffle:
            random.shuffle(files)

        for file in files:
            yield ProSolDataset(torch.load(file, map_location = self.device))
                
class ProSolDataset(Dataset):
    """
    Dataset class for problem-solution pairs.

    Args:
        pro_sol_pairs (dict): Dictionary containing 'problem' and 'solution' keys.
            The 'problem' key should have a value of type dict, containing 'info' and 'tensor' keys.
            The 'solution' key should have a value of type list.

    """

    def __init__(self, pro_sol_pairs):
        assert 'problem' in pro_sol_pairs and 'solution' in pro_sol_pairs, \
            "Input dictionary must contain 'problem' and 'solution' keys"
        
        self.problems = pro_sol_pairs['problem']
        self.solutions = pro_sol_pairs['solution']
        
        assert type(pro_sol_pairs['problem']) == dict and type(pro_sol_pairs['solution']) == list, \
                "Data must be in the form of {'problem': {'info':[], 'tensor':[]}, 'solution': []}"
                
        assert len(self.problems['tensor']) == len(self.solutions) and \
            len(self.problems['info']) == len(self.solutions), \
                "Number of problems and solutions must be equal"

    def __len__(self):
        return len(self.solutions)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._get_slice(idx)
        elif isinstance(idx, int):
            if idx >= len(self) or idx < 0:
                raise IndexError("Index out of range")
            return self._get_item(idx)
        else:
            raise TypeError("Index must be an integer or a slice")

    def _get_item(self, idx):
        problem_data = {k: v[idx] for k, v in self.problems.items()}
        solution_data = self.solutions[idx]
        return problem_data, solution_data

    def _get_slice(self, s):
        sliced_problems = {k: v[s] for k, v in self.problems.items()}
        sliced_solutions = self.solutions[s]
        return sliced_problems, sliced_solutions
    
def collate_fn(batch):

    problems_batch = {key: [] for key in batch[0][0]}
    solutions_batch = []

    for problems, solutions in batch:
        for key, value in problems.items():
            problems_batch[key].append(value)
        solutions_batch.append(solutions)

    for key in problems_batch:
        problems_batch[key] = problems_batch[key]

    solutions_batch = solutions_batch

    return problems_batch, solutions_batch

    
if __name__ == '__main__':
    config = None
    dir_dict = 'IL_test/0605/'
    dataset = ILDataScheduler(config, dir_dict, reload = True)

    dataset.save_processed_data(classify_mas=True)

