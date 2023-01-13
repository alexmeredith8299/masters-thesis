"""
This module is used to download saved models from the sftp server.
"""
import pysftp
import textwrap
import os
import argparse

def parse_args():
    """
    Parse command line arguments specifiyng the models to generate sbatch files for.

    Returns 
    --------
        args: argparse.Namespace
            parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Download files from the SLURM cluster', formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=textwrap.dedent('''\
        Example usage:
            python3 get_from_slurm.py -p password -m 08-09 -d /Users/alexmeredith -f .csv 995

        The example above will open an sftp connection to the SLURM cluster using `password`
        as the password, and will download all .csv files and files with `995` in the filename
        from any folders that match the pattern `08-09`. The files will be downloaded to the
        local machine at `/Users/alexmeredith`.
        '''))

    #Server-related arguments
    parser.add_argument('-n', '--hostname', default='eofe7.mit.edu', type=str, help='sftp server hostname (default: eofe7.mit.edu)')
    parser.add_argument('-u', '--username', default='ameredit', type=str, help='username for the sftp server (default: ameredit)')
    parser.add_argument('-p', '--password', default=None, type=str, help='password for the sftp server', required=True)

    #Describes files to get
    parser.add_argument('-r', '--root_path', default='/pool001/ameredit/saved_models', type=str, help='path to the root folder on the sftp server (default: /pool001/ameredit/saved_models)')
    parser.add_argument('-f', '--file_match', nargs='+', help='get files that match the given pattern')
    parser.add_argument('-l', '--latest', action='store_true', help='get the latest epoch that is saved for all models (actually gets second-latest, because latest save seems to be corrupted sometimes)')#TODO use
    parser.add_argument('-m', '--folder_match', nargs='+', help='get files from folders that match the given pattern')

    #Describes where to put files
    parser.add_argument('-d', '--dest_path', default='/Users/alexmeredith/masters-thesis/cloud-detection-code/scripts/saved_models', type=str, help='path to the destination folder on the local machine\n (default: /Users/alexmeredith/masters-thesis/cloud-detection-code/scripts/saved_models)')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing files in the destination folder')


    args = parser.parse_args()

    return args

def list_folders_for_download(sftp, root_path, folder_match):
    """
    List folders on the sftp server that match the given pattern(s).

    Arguments
    ----------
        sftp: pysftp.Connection
            sftp connection to the server
        root_path: str
            path to the root folder on the sftp server
        folder_match: list of str
            list of patterns to match folders against

    Returns
    ----------
        folders_to_get: list of str
            list of folders on the sftp server that match the given pattern(s)
    """
    folders_to_get = []

    with sftp.cd(root_path):
        file_list = sftp.listdir()
        for exp_folder in file_list:
            if exp_folder[0:3] == 'exp':
                exp_file_list = sftp.listdir(exp_folder)
                for folder in exp_file_list:
                    for pattern in folder_match:
                        if pattern in folder and os.path.join(exp_folder, folder) not in folders_to_get:
                            folders_to_get.append(os.path.join(exp_folder, folder))
    return folders_to_get

def avoid_overwrite(dest_path, folders_to_get):
    """
    Get list of folders to download from that will not overwrite existing files.

    Arguments
    ----------
        dest_path: str
            path to the destination folder on the local machine
        folders_to_get: list of str
            list of folders on the sftp server that match the given pattern(s)

    Returns
    ----------
        folders_no_overwrite: list of str
            list of folders on the sftp server that will not overwrite existing files
    """
    folders_no_overwrite = []
    for folder in folders_to_get:
        if not os.path.exists(os.path.join(dest_path, folder)):
            folders_no_overwrite.append(folder)
    return folders_no_overwrite

def get_latest_pattern(sftp, root_path, folders_to_get):
    """
    Loop through folders on sftp server and add latest (and second-latest)
    epoch available in all folders to the list of patterns to download.

    Arguments 
    ----------
        sftp: pysftp.Connection
            sftp connection to the server
        root_path: str
            path to the root folder on the sftp server
        folders_to_get: list of str 
            list of folders on the sftp server to download files from
    """
    epoch_sets = []
    for folder in folders_to_get:
        with sftp.cd(os.path.join(root_path, folder)):
            file_list = sftp.listdir()
            epoch_set = set()
            for file in file_list:
                if file[-3:] == '.pt':
                    epoch = int(file.split('_')[-1].split('.')[0])
                    epoch_set.add(epoch)
            epoch_sets.append(epoch_set)
    epoch_intersection = set().union(*epoch_sets)
    latest_epoch = max(epoch_intersection)
                   

def download_files(sftp, root_path, dest_path, folders_to_get, file_match, overwrite=False):
    """
    Download files from the sftp server to the local machine.

    Arguments
    ----------
        sftp: pysftp.Connection
            sftp connection to the server
        root_path: str
            path to the root folder on the sftp server
        dest_path: str
            path to the destination folder on the local machine
        folders_to_get: list of str
            list of folders on the sftp server to download files from 
        file_match: list of str
            list of patterns to match files against
    """
    for folder in folders_to_get:
        with sftp.cd(os.path.join(root_path, folder)):
            file_list = sftp.listdir()
            #Only download files that match the given pattern(s)
            for file in file_list:
                for pattern in file_match:
                    if pattern in file:
                        #Make sure destination folder exists
                        if not os.path.exists(os.path.join(dest_path, folder)):
                            os.makedirs(os.path.join(dest_path, folder))
                        #Download file
                        if not os.path.exists(os.path.join(dest_path, folder, file)) or overwrite:
                            sftp.get(file, os.path.join(dest_path, folder, file))

def main(args):
    """
    Main function. Opens an sftp connection to the server, downloads the files, and closes the connection.

    Arguments
    ----------
        args: argparse.Namespace
            parsed command line arguments
    """
    #Connect to sftp server
    sftp = pysftp.Connection(args.hostname, username=args.username, password=args.password)

    #Download files
    folders_to_get = list_folders_for_download(sftp, args.root_path, args.folder_match)
    #if not args.overwrite:
        #folders_to_get = avoid_overwrite(args.dest_path, folders_to_get)
    download_files(sftp, args.root_path, args.dest_path, folders_to_get, args.file_match, overwrite=args.overwrite)

    #Close connection
    sftp.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)

