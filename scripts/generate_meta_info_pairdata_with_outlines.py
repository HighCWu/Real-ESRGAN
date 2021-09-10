import argparse
import glob
import os


def main(args):
    txt_file = open(args.meta_info, 'w')
    img_paths_gt = sorted(glob.glob(os.path.join(args.input[0], '*')))
    img_paths_lq = sorted(glob.glob(os.path.join(args.input[1], '*')))
    img_paths_ol = sorted(glob.glob(os.path.join(args.input[2], '*')))

    assert len(img_paths_gt) == len(img_paths_lq) == len(img_paths_ol), ('GT folder, LQ folder and OL folder should have the same length, but got '
                                                    f'{len(img_paths_gt)}, {len(img_paths_lq)} and {len(img_paths_ol)}.')

    for img_path_gt, img_path_lq, img_path_ol in zip(img_paths_gt, img_paths_lq, img_paths_ol):
        img_name_gt = os.path.relpath(img_path_gt, args.root[0])
        img_name_lq = os.path.relpath(img_path_lq, args.root[1])
        img_name_ol = os.path.relpath(img_path_ol, args.root[2])
        print(f'{img_name_gt}, {img_name_lq}, {img_name_ol}')
        txt_file.write(f'{img_name_gt}, {img_name_lq}, {img_name_ol}\n')


if __name__ == '__main__':
    """Generate meta info (txt file) for paired images with outlines.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['datasets/DF2K/DIV2K_train_HR_sub', 'datasets/DF2K/DIV2K_train_LR_bicubic_X4_sub', 'datasets/DF2K/DIV2K_train_HR_sub_outlines'],
        help='Input folder, should be [gt_folder, lq_folder, ol_folder]')
    parser.add_argument('--root', nargs='+', default=[None, None, None], help='Folder root, will use the ')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='datasets/DF2K/meta_info/meta_info_DIV2K_sub_pair_with_outlines.txt',
        help='txt path for meta info')
    args = parser.parse_args()

    assert len(args.input) == 3, 'Input folder should have two elements: gt folder, lq folder and ol_folder'
    assert len(args.root) == 3, 'Root path should have two elements: root for gt folder, lq folder and ol_folder'
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    for i in range(3):
        if args.input[i].endswith('/'):
            args.input[i] = args.input[i][:-1]
        if args.root[i] is None:
            args.root[i] = os.path.dirname(args.input[i])

    main(args)
