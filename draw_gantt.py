import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import random
import os
import numpy as np

def draw_gantt(actions, num_jobs, num_machines, matrix_proc_time, folder='./', suffix = ''):
    """
    Draw a Gantt chart based on the given actions, number of jobs, number of machines, and processing time matrix.

    Parameters:
    - actions (list): A list of tuples representing the actions performed, where each tuple contains the operation, machine, and job.
    - num_jobs (int): The total number of jobs.
    - num_machines (int): The total number of machines.
    - matrix_proc_time (list): A matrix representing the processing time for each operation on each machine.
    - folder (str): The output folder where the Gantt chart image will be saved. Default is the current directory.
    - suffix (str): A suffix to be added to the output image filename. Default is an empty string.

    Returns:
    - output_file (str): The path to the saved Gantt chart image file.
    - operation_times (dict): A dictionary containing the start and end times for each operation.

    """
    # Rest of the code...
def draw_gantt(actions, num_jobs, num_machines, matrix_proc_time, folder='./', suffix = ''):

    if not os.path.exists(folder):
        os.makedirs(folder)

    machine_end_times = {machine: 0 for machine in range(num_machines)}
    job_end_times = {job: 0 for job in range(num_jobs)}
    job_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(num_jobs)]

    fig, ax = plt.subplots()

    operation_times = {}

    for operation, machine, job in actions:
        if machine == -1:
            continue
        if matrix_proc_time[operation][machine] == 0:
            continue

        start_time = max(machine_end_times[machine], job_end_times[job])
        end_time = start_time + matrix_proc_time[operation][machine]

        machine_end_times[machine] = end_time
        job_end_times[job] = end_time

        operation_times[operation] = (start_time, end_time)

        rect = patches.Rectangle((start_time, machine), end_time - start_time, 0.8, edgecolor='black', facecolor=job_colors[job])
        ax.add_patch(rect)
        ax.text(start_time + (end_time - start_time) / 2, machine + 0.4, f'Op{operation+1}', ha='center', va='center')

    makespan = max(machine_end_times.values())

    ax.axvline(x=makespan, color='r', linestyle='--', label='Makespan')
    ax.text(np.ceil(1.05 * makespan), 0.5, f'Makespan: {int(makespan)}', color='red', rotation=90)

    ax.set_ylabel('Machine')
    ax.set_xlabel('Time')
    ax.set_title('Gantt Chart')
    ax.set_yticks([i + 0.4 for i in range(num_machines)])
    ax.set_yticklabels([f'Machine {i+1}' for i in range(num_machines)])
    ax.set_ylim(-0.1, num_machines)
    ax.set_xlim(0, np.ceil(makespan * 1.1) + 1)  # Add some space for readability

    legend_patches = [patches.Patch(color=job_colors[i], label=f'Job {i+1}') for i in range(num_jobs)]
    legend_patches.append(patches.Patch(color='red', label='Makespan', linestyle='--'))
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    output_file = os.path.join(folder, f'gantt_chart{str(suffix)}.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    return output_file, operation_times

def draw_sol_gantt(sol, folder='./', suffix = '', show = False):
    if isinstance(sol, str):
        with open(sol, 'r') as file:
            lines = file.read().splitlines()
    else:
        lines = sol
    makespan = int(float(lines[0]))
    proc_matrix = np.array([list(map(int, map(float, line.split()))) for line in lines[1:]])

    process_id = proc_matrix[:, 0]
    machine_id = proc_matrix[:, 1]
    job_id = proc_matrix[:, 2]
    start_time = proc_matrix[:, 3]
    end_time = proc_matrix[:, 4]

    unique_jobs = list(set(job_id))
    colors = plt.cm.get_cmap('tab20', len(unique_jobs))
    job_color_map = {job: colors(i) for i, job in enumerate(unique_jobs)}

    fig, ax = plt.subplots(figsize=(15, 10))

    for i in range(len(process_id)):
        ax.broken_barh([(start_time[i], end_time[i] - start_time[i])], 
                    (machine_id[i] * 10, 9), 
                    facecolors=(job_color_map[job_id[i]]),
                    edgecolor='black')
        ax.text(start_time[i] + (end_time[i] - start_time[i]) / 2, machine_id[i] * 10 + 5, str(process_id[i]),
                va='center', ha='center', color='black', fontsize=8)

    patches = [mpatches.Patch(color=job_color_map[job], label=f'Job {job}') for job in unique_jobs]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.axvline(x=makespan, color='r', linestyle='--', label='Makespan')
    ax.text(np.ceil(1.01 * makespan), 0.5, f'Makespan: {int(makespan)}', color='red', rotation=90)

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine ID')
    ax.set_title('Gantt')
    ax.set_yticks([i * 10 + 5 for i in range(len(set(machine_id)))])
    ax.set_yticklabels([f'Ma {i}' for i in set(machine_id)])

    output_file = os.path.join(folder, f'gantt_chart{str(suffix)}.png')
    if show:
        plt.show()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
