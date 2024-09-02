import os


if __name__ == '__main__':
    root_path = input('请输入需要导出会议所在的根目录：')
    meeting_name = os.listdir(root_path)

    save_file = os.path.join(root_path, 'save_file.txt')
    with open(save_file, 'w', encoding='utf-8') as txt_save:
        for meeting in meeting_name:
            txt_save.write(meeting)
            txt_save.write('\n')

        txt_save.close()
