from collections import defaultdict
import numpy as np
def shrink_schedule(actions, num_jobs, num_machines, matrix_proc_time, ignore_supernode = True, return_sol = False):
    '''
    This function is used to shrink the schedule to a smaller makespan.
    Specifically, it has two functions:
    - Ignore the supernode: if the processing time of an operation on a machine is 0, then ignore this operation (optional).
    - Shrink the schedule: find a better start time for each operation on each machine by searching idle time intervals supporting 
        earlier processing.
    
    '''
    machine_end_times = {machine: 0 for machine in range(num_machines)}
    job_end_times = {job: 0 for job in range(num_jobs)}
    operation_times = defaultdict(lambda: (0,0))
    machine_start_pointer = defaultdict(list)
    end_start_pointer = defaultdict(lambda: [0])

    for operation, machine, job in actions:
        if machine == -1:
            continue
        proc_time = matrix_proc_time[operation][machine]
        if proc_time == 0:
            if ignore_supernode:
                continue
        if len(machine_start_pointer[machine]) == 0:
            start_time = job_end_times[job]
            end_time = start_time + proc_time
            machine_start_pointer[machine].append(start_time)
            end_start_pointer[machine].append(end_time)
        else:
            start_time_np = np.array(machine_start_pointer[machine])
            end_time_np = np.array(end_start_pointer[machine][:-1])
            idle_interval = start_time_np - end_time_np
            assert (idle_interval>=0).all()
            com = (idle_interval >= proc_time) & (start_time_np >= job_end_times[job] + proc_time)
            if np.sum(com) == 0:
                start_time = max(end_start_pointer[machine][-1], job_end_times[job])
                end_time = start_time + proc_time
                machine_start_pointer[machine].append(start_time)
                end_start_pointer[machine].append(end_time)
            else:
                index = np.argmax(com)
                start_time = max(end_time_np[index], job_end_times[job])
                end_time = start_time + proc_time
                machine_start_pointer[machine].insert(index, start_time)
                end_start_pointer[machine].insert(index + 1, end_time)

        machine_end_times[machine] = end_start_pointer[machine][-1]
        job_end_times[job] = end_time
        operation_times[operation] = (start_time, end_time)
    # print(operation_times)
    if not return_sol:
        return max(machine_end_times.values())
    
    anwser = [str(max(machine_end_times.values()))]
    for action in actions:
        if action[1] == -1:
            continue
        anwser.append(' '.join(map(str, action))+f' {operation_times[action[0]][0]} {operation_times[action[0]][1]}')
    return max(machine_end_times.values()), anwser
