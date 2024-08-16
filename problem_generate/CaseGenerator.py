import os
import random
import re
import pickle
class CaseGenerator:
    '''
    IPPS instance generator
    '''
    def __init__(self, job_num,save=False):
        self.job_num = job_num
        self.save=save


    def get_case(self,case_num):
        '''
        Generate IPPS instance
        '''
        while True:
            job_pool = list(range(1, 19))
            input_jobs = random.sample(job_pool, self.job_num)
            input_jobs=sorted(input_jobs)
            current_offset = 0  
            combined_out = []  
            combined_in = []
            updated_info_lines = []
            machine = set()

            for job_index in input_jobs:
                machine = machine | self.machine_job[str(job_index)]
                with open(os.path.join('jobs', f'job{job_index}.txt'), 'r') as file:  # 使用相对路径构建文件路径
                    lines = file.readlines()

                    # 找到out部分的开始和结束
                    out_start_index = lines.index("out\n") + 1
                    out_end_index = lines.index("in\n", out_start_index)

                    # 处理out部分的每一行
                    for line in lines[out_start_index:out_end_index]:
                        if line.strip():  
                            new_line = offset_line(line, current_offset)
                            combined_out.append(new_line)

                    # 找到in部分的开始和结束
                    in_start_index = lines.index("in\n") + 1
                    in_end_index = lines.index("info\n", in_start_index)

                    # 处理in部分的每一行
                    for line in lines[in_start_index:in_end_index]:
                        if line.strip(): 
                            new_line = offset_line(line, current_offset)
                            combined_in.append(new_line)

                    # 找到info部分的开始
                    info_start_index = lines.index("info\n") + 1

                    # 处理info部分的每一行
                    for line in lines[info_start_index:]:
                        if line.strip(): 
                            parts = line.split()
                            parts[0] = str(int(parts[0]) + current_offset)
                            updated_info_lines.append(' '.join(parts))

                    # 更新偏移量
                    current_offset += int(lines[0].split()[2])

            # 如果机器数量不等于 15，则重新选择作业
            if len(machine) != 15:
                continue

            new_line=[f"{len(input_jobs)} {len(machine)} {current_offset}\n"]
            new_line.append("out\n")
            for line in combined_out:
                new_line.append(line)
            new_line.append("in\n")
            for line in combined_in:
                new_line.append(line)
            new_line.append("info\n")
            for line in updated_info_lines:
                new_line.append(line +"\n")
            if self.save==True:
                os.makedirs("problem", exist_ok=True)
                with open(os.path.join('problem', f'problem_job_{"_".join(map(str, input_jobs))}.txt'), 'w') as file: 
                    file.writelines(new_line)
            

            return [line.replace('\n', '')for line in new_line]


def offset_line(line, offset):
    
    def offset_match(match):
 
        return str(int(match.group()) + offset)
    
 
    updated_line = re.sub(r'\b\d+\b', offset_match, line)

    def correct_parentheses(match):
        # Match group 1 is the content inside parentheses
        inside_parentheses = match.group(1)
        # Correct the offset by subtracting it once
        corrected = str(int(inside_parentheses) - offset)
        return f"({corrected})"

    updated_line = re.sub(r'\((\d+)\)', correct_parentheses, updated_line)
    return updated_line
if __name__ == "__main__":
    os.makedirs("problem", exist_ok=True)
    with open("machine_job.pkl", "rb") as f:
        machine_job = pickle.load(f)
    case_generator = CaseGenerator(job_num=5)
    lines = case_generator.get_case()

 
