# [YOLOv4手把手實作應用](https://suyenting.github.io/post/yolov4-hands-on/)
# [How to save command output to file](https://www.windowscentral.com/how-save-command-output-file-using-command-prompt-or-powershell)
# [從路徑中獲取沒有副檔名的檔名](https://www.delftstack.com/zh-tw/howto/python/python-get-filename-without-extension-from-path/#%e5%9c%a8-python-%e4%b8%ad%e4%bd%bf%e7%94%a8-pathlibpathstem-%e6%96%b9%e6%b3%95%e5%be%9e%e8%b7%af%e5%be%91%e4%b8%ad%e7%8d%b2%e5%8f%96%e6%b2%92%e6%9c%89%e5%89%af%e6%aa%94%e5%90%8d%e7%9a%84%e6%aa%94%e5%90%8d)
# 需放在與darknet.exe同層的資料夾

import os
import sys
from pathlib import Path


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_filename_without_extension(path):
    return Path(path).stem


def main():
    # 檢查參數是否足夠
    if len(sys.argv) != 4:
        print("Usage: python script.py <modelCfg> <modelWeights> <imageFilePath>")
        sys.exit(1)

    # 設定參數
    # 接收參數-預測照片路徑txt檔案
    # 例如:
    # modelCfg = 'cfg/yolo-obj.cfg'
    # modelWeights = 'backup/weights/yolov4.weights'
    # imageFilePath = 'customdata/ColumbiaDogsDataset/testingv7.txt'
    # path = "C:/Proposal/Stanford_Dogs_Dataset"
    # os.chdir(path)
    # systeminfo > C:/Proposal/Stanford_Dogs_Dataset/output.txt 儲存cmd 結果

    modelCfg = sys.argv[1]
    modelWeights = sys.argv[2]
    imageFilePath = sys.argv[3]

    output_dirname = 'kitty'
    thres = 0.5

    image_dir = 'C:/Proposal/darknet/build/darknet/x64/' + output_dirname
    info_dir = 'C:/Proposal/Info/original/' + output_dirname

    # 建立預測資料夾與資訊資料夾
    create_directory(image_dir)
    create_directory(info_dir)

    # 讀取照片路徑
    imageFiles = []
    with open(imageFilePath, 'r') as myFile:
        imageFiles = [line.strip() for line in myFile]

    # 迴圈執行darknet預測執行
    for imageFile in imageFiles:
        # 從路徑中獲取沒有副檔名的檔名
        commands_txt_path = get_filename_without_extension(imageFile) + '_info.txt'

        # 建立cmd指令
        commands = f'darknet detector test C:/Proposal/darknet/build/darknet/x64/data/kitty.data {modelCfg} {modelWeights} {imageFile} -thresh {thres} -ext_output -dont_show > {commands_txt_path}'

        # 執行cmd指令
        os.system(commands)

        # 將預測照片重新命名並移至預測資料夾下
        os.replace('predictions.png', os.path.join(output_dirname, os.path.basename(imageFile)))

        new_path = os.path.join(info_dir, commands_txt_path)
        os.rename(commands_txt_path, new_path)


if __name__ == "__main__":
    main()
