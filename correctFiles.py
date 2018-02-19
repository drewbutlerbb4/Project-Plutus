"""
Removes unwanted rows from every CSV file found in the given list
An unwanted row is any row that contains no information (in this case, it would only have a date)
The reformatted files will be placed in a specified folder


KNOWN BUGS:
1.) If the files are saved on a Windows Machine then any instrument with any of the following names
    will not be allowed as they are reserved keywords
    Reserved Keywords: CON,PRN,AUX,NUL,COM1,COM2,COM3,COM4,COM5,COM6,COM7,COM8,COM9,LPT1,LPT2,LPT3,
                        LPT4,LPT5,LPT6,LPT7,LPT8,LPT9
    Resolutions: There is a strip of code marked BUG1 RESOLUTION in reformat_csv() that can be used
                 if any of the instruments were named a reserved keyword
    Notes:       This resolution is not implemented in an attempt to allow for more efficient scaling

author: Andrew Butler
"""
from pathlib import Path


def reformat_csv(instr_name, csv_common_path, to_folder):
    """
    Reformats a CSV file by getting rid of all unnecessary cells



    :param instr_name:          The name of the instrument for the file that is being reformatted
    :param csv_common_path:     The path to the folder containing the csv files
    :param to_folder:           The path of the folder where the new files will be downloaded
    """

    if instr_name == "CON":
        instr_name = "_CON"

    read_path = csv_common_path+"\\"+instr_name+".L.csv"
    write_path = to_folder+"\\"+instr_name+".L.csv"
    path_help = Path(read_path)

    if path_help.is_file():
        read_file = open(read_path)
        write_file = open(write_path, "w+")

        cur_line = read_file.readline()

        while cur_line != "":
            cells = cur_line.split(",")
            num_nullcells = 0

            for cell in cells:
                if cell == "null":
                    num_nullcells += 5
            if not (num_nullcells >= 6):
                line_string = ""
                for cell in cells:
                    line_string = line_string + "," + cell
                line_string = line_string[1:-1]
                write_file.write(line_string)
            cur_line = read_file.readline()
    else:
        print("Failed to find: " + instr_name)


def check_file_list(list_path, csv_common_path, to_folder):
    """
    Scans a file for csv files that need to be reformatted and then reformats the files specified

    :param list_path:           The name of the file of the list of instruments
    :param csv_common_path:     The path to the folder containing the csv files
    :param to_folder:           The path to the folder where the files will be downloaded to
    """
    read_file = open(list_path)

    cur_instr = read_file.readline()

    while cur_instr != "":
        cells = cur_instr.split(",")

        if len(cells) > 0:
            reformat_csv(cells[0], csv_common_path, to_folder)
        cur_instr = read_file.readline()


path = "D:\\MarketEval\\InstrumentValueByDay\\InstrumentsLSEStripped - Sheet1.csv"
csv_path = "C:\\Users\\Andrew Butler\\Downloads"
dl_folder = "D:\\MarketEval\\ReformattedCSV"
check_file_list(path, csv_path, dl_folder)
