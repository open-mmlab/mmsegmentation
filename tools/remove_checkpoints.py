import os

work_dirs_path = "work_dirs"
iter_to_remove = "1500"
for project_name in os.listdir(work_dirs_path):
    project_path = os.path.join(work_dirs_path, project_name)
    files_to_remove = [file_name for file_name in os.listdir(project_path) if iter_to_remove in file_name and ".pth" in file_name]
    for file_to_remove in files_to_remove:
        os.remove(os.path.join(project_path, file_to_remove))       