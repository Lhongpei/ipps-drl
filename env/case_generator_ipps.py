import os
import random
import re
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from problem_generate.Jobs_Generator import jobs_generator
from tqdm import tqdm
class CaseGenerator:
    '''
    IPPS instance generator
    '''
    def __init__(self, job_num=5, machine_num=15, if_new_job = False, job_folder = None):
        self.job_num = job_num
        self.machine_num = machine_num
        self.if_new_job = if_new_job
        self.job_folder = job_folder

    def get_case_with_new_job(self,save=False,new_folder=None,mas_p=0.5,or_p=0.3,ope=True,ope_num=10, ope_range=(5,20),total_ope_range=(5,100),time_range=(5,50),max_out=2,road_num=3, ope_num_orpath=3,and_road_num=3,ope_num_andpath=3,and_p=0.3,path=None):
        
        job_list=[]


        while True:
            for i in range(self.job_num):
                new_job=jobs_generator(machine_range=(self.machine_num,self.machine_num),mas_p=mas_p,or_p=or_p,ope=ope,ope_num=ope_num, ope_range=ope_range,total_ope_range=total_ope_range,time_range=time_range,max_out=max_out,road_num=road_num, ope_num_orpath=ope_num_orpath,and_road_num=and_road_num,ope_num_andpath=ope_num_andpath,and_p=and_p,path=f'./train_tmp/graph',save=True)
                job_list.append(new_job)
                
            # 列出文件夹中的所有文件并按索引选择

            job_pool = list(range(len(job_list)))
            input_jobs = random.sample(job_pool, self.job_num)
            
            current_offset = 0
            combined_out = []
            combined_in = []
            updated_info_lines = []
            all_machines_used = set()

            for job_index in input_jobs:


                lines = job_list[job_index]

                # 找到out部分的开始和结束
                out_start_index = lines.index("out") + 1
                out_end_index = lines.index("in", out_start_index)

                # 处理out部分的每一行
                for line in lines[out_start_index:out_end_index]:
                    if line.strip():  
                        new_line = offset_line(line, current_offset)
                        combined_out.append(new_line)

                # 找到in部分的开始和结束
                in_start_index = lines.index("in") + 1
                in_end_index = lines.index("info", in_start_index)

                # 处理in部分的每一行
                for line in lines[in_start_index:in_end_index]:
                    if line.strip(): 
                        new_line = offset_line(line, current_offset)
                        combined_in.append(new_line)

                # 找到info部分的开始
                info_start_index = lines.index("info") + 1

                # 处理info部分的每一行
                for line in lines[info_start_index:]:
                    if line.strip(): 
                        parts = line.split()
                        parts[0] = str(int(parts[0]) + current_offset)
                        updated_info_lines.append(' '.join(parts))
                        machines_used = parts[2::2]  # 每隔两个元素获取机器编号
                        all_machines_used.update(map(int, machines_used))

                # 更新偏移量
                current_offset += int(lines[0].split()[2])

            # 如果机器数量不等于 machine_num，则重新选择作业
            if all_machines_used != set(range(1, self.machine_num + 1)):
                continue

            new_line = [f"{len(input_jobs)} {len(all_machines_used)} {current_offset}\n"]
            new_line.append("out\n")
            for line in combined_out:
                new_line.append(line + "\n")
            new_line.append("in\n")
            for line in combined_in:
                new_line.append(line + "\n")
            new_line.append("info\n")
            for line in updated_info_lines:
                new_line.append(line + "\n")
            if save:
                os.makedirs(new_folder, exist_ok=True)
                file_name = path+'.txt'
                with open(os.path.join(new_folder, file_name), 'w') as file: 
                    file.writelines(new_line)
            
            return [line.replace('\n', '') for line in new_line]
        
    def get_case_with_job_files(self,job_folder,save=False,new_folder=None):
        '''
        Generate IPPS instance
        '''
        if not os.path.exists(job_folder):
            raise FileNotFoundError(f"Folder {job_folder} does not exist")

        while True:

            # 列出文件夹中的所有文件并按索引选择
            index=[]
            job_files = [f for f in os.listdir(job_folder) if f.endswith('.txt') and f"mas_{self.machine_num}" in f]
            
            # 确保文件按升序排列
            job_files.sort()
            
            # 获取文件数量范围
            job_pool = list(range(len(job_files)))
            
            # 随机选择 job_num 个文件
            input_jobs = random.sample(job_pool, self.job_num)
            
            # 按升序排序
            input_jobs = sorted(input_jobs)
            
            # 返回选择的文件名列表
            
            current_offset = 0
            combined_out = []
            combined_in = []
            updated_info_lines = []
            all_machines_used = set()

            for job_index in input_jobs:
                
                job_file=job_files[job_index]
                index.append(int(re.search(r'job_(\d+)_mas', job_file).group(1)))
                with open(os.path.join(job_folder, job_file), 'r') as file:
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
                            machines_used = parts[2::2]  # 每隔两个元素获取机器编号
                            all_machines_used.update(map(int, machines_used))

                    # 更新偏移量
                    current_offset += int(lines[0].split()[2])

            # 如果机器数量不等于 machine_num，则重新选择作业
            if all_machines_used != set(range(1, self.machine_num + 1)):
                continue

            new_line = [f"{len(input_jobs)} {len(all_machines_used)} {current_offset}\n"]
            new_line.append("out\n")
            for line in combined_out:
                new_line.append(line)
            new_line.append("in\n")
            for line in combined_in:
                new_line.append(line)
            new_line.append("info\n")
            for line in updated_info_lines:
                new_line.append(line + "\n")
            if save:
                os.makedirs(new_folder, exist_ok=True)
                file_name = f'{self.job_num}_{self.machine_num}_problem_job_{"_".join(map(str, index))}.txt'
                with open(os.path.join(new_folder, file_name), 'w') as file: 
                    file.writelines(new_line)
            
            return [line.replace('\n', '') for line in new_line]
        
    def get_case(self):
        if self.if_new_job:
            return self.get_case_with_new_job()
        else:
            return self.get_case_with_job_files(self.job_folder)

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
    
    # with open("machine_job.pkl", "rb") as f:
    #     machine_job = pickle.load(f)
    for i in tqdm(range(100)):
        job_num=6
        machine_num=5
        case_generator = CaseGenerator(job_num,machine_num)
        lines = case_generator.get_case_with_job_files('env/job_with_mas_5',save=True,new_folder='or_problem')
        print(lines)




 
