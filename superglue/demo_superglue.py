#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

# def extract_frames(video_path, output_path, num_frames=90):
#     vidcap = cv2.VideoCapture(video_path)
#     total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frames_to_skip = total_frames // num_frames

#     success,image = vidcap.read()
#     count = 0
#     frame_number = 0

#     while success:
#         if frame_number % frames_to_skip == 0:
#             cv2.imwrite(f"{output_path}/frame{count:04d}.jpg", image)
#             count += 1
#         success,image = vidcap.read()
#         frame_number += 1

#         if count == num_frames:
#             break

#     vidcap.release()

# def images_to_video(image_folder, output_path, fps=30):
#     image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
#     image_files.sort()

#     if not image_files:
#         print(f"No '.png' files found in {image_folder}.")
#         return

#     # 使用 mp4v 編碼器
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#     # 讀取第一張圖片的尺寸
#     img = cv2.imread(os.path.join(image_folder, image_files[0]))
#     height, width, layers = img.shape

#     # 設定 VideoWriter
#     video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     for image_file in image_files:
#         img = cv2.imread(os.path.join(image_folder, image_file))
#         video.write(img)

#     video.release()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    # # 把無人機影片切成圖片
    # video_file = 'D:/image_experience/SuperGluePretrainedNetwork-master/2023_11_23_13_59_06.mp4'
    # output_folder = 'freiburg_sequence'
    # extract_frames(video_file, output_folder, num_frames=90)
    
    #圖像調整大小
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    #檢查運行環境
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    
    #模型配置
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']
    
    #處理影像資訊
    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()    #獲取下一幀。frame:當前幀影像數據；ret:是否成功讀取此幀
    assert ret, 'Error when reading the first frame (try different --input?)'#失敗則回傳錯誤訊息

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    image_confidences = []
    all_average_confidences = []
    all_max_confidences = []
    average_max05_confidence = []
    average_max05_append_confidence = []
    average_max06_confidence = []
    average_max06_append_confidence = []
    count_all = 0
    count_above_05_total = 0
    count_above_06_total = 0
    
    
    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished demo_superglue.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1

        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        timer.update('forward')

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

        if not opt.no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold']+d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints

        timer.update('viz')
        timer.print()
    
        if opt.output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)
            
            
            #---------------------------
            image_confidences.extend(confidence[valid])  #Append confidence scores
            
            #print出信心度
            print(f'Confidence values for the current frame:{image_confidences}')
            
            #print出每張圖平均信心度
            average_confidence = np.mean(image_confidences) if image_confidences else 0.0
            print(f"Average confidence: {average_confidence:.4f}")
            
            #計算所有圖平均信心度
            all_average_confidences.append(average_confidence)
            
            #-----------------------------------------------------------
            #計算最大值的線
            max_confidence = np.max(image_confidences) if image_confidences else 0.0
            print(f"each max confidence: {max_confidence:.4f}")
            all_max_confidences.append(max_confidence)
            
            #計算及print出大於0.5的值
            image_confidences_array = np.array(image_confidences)
            filtered_values = image_confidences_array[image_confidences_array > 0.5]
            print(f"Values greater than 0.5:", filtered_values)
            
            # filtered_values_6 = image_confidences_array[image_confidences_array > 0.6]
            # print(f"Values greater than 0.6:", filtered_values_6)
            
            #計算及print出大於0.5的次數
            count_above_05 = np.count_nonzero(filtered_values > 0.5)
            print(f"Number of values greater than 0.5: {count_above_05}")
            count_above_05_total += count_above_05
            
            # count_above_06 = np.count_nonzero(filtered_values > 0.6)
            # print(f"Number of values greater than 0.6: {count_above_06}")
            # count_above_06_total += count_above_06
            
            #計算所有匹配線數量
            count = np.count_nonzero(image_confidences_array)
            count_all += count
            
            #計算及print出單張圖大於0.5的平均值
            average_max05_confidence = np.mean(filtered_values)
            # average_max06_confidence = np.mean(filtered_values_6)
            
            if np.isnan(average_max05_confidence):
                print("Average confidence for values greater than 0.5 is not a valid numerical value.")
            else:
                #單張圖大於0.5的匹配線平均值
                print(f"Average confidence for values greater than 0.5: {average_max05_confidence}")
                average_max05_append_confidence.append(average_max05_confidence)
            
            # if np.isnan(average_max06_confidence):
            #     print("Average confidence for values greater than 0.6 is not a valid numerical value.")
            # else:
            #     #單張圖大於0.6的匹配線平均值
            #     print(f"Average confidence for values greater than 0.6: {average_max06_confidence}")
            #     average_max06_append_confidence.append(average_max06_confidence)
            
            
            image_confidences = []    #清空(讓下一次不會繼承此次結果)
            
            
        print('-------------------------------------------------------------------------------------------------------------------------------')
    
    #所有圖的平均信心度
    overall_average_confidence = np.mean(all_average_confidences) if all_average_confidences else 0.0
    print(f"Overall average confidence across all images: {overall_average_confidence:.4f}")
    
    #最大信心度的圖(frame)
    overall_frame_max_confidence = np.max(all_average_confidences) if all_average_confidences else 0.0
    print(f"Maximum_frame_confidence: {overall_frame_max_confidence:.4f}")
    
    #最大信心度的匹配線(line)
    overall_max_confidences = np.max(all_max_confidences) if all_max_confidences else 0.0
    print(f"Maximum_line_confidence: {overall_max_confidences:.4f}")
    
    #所有圖大於0.5的匹配線平均值
    # overall_max05_confidences = np.mean(average_max05_append_confidence) if average_max05_append_confidence else 0.0
    # print(f"Maximum_05_average_confidence: ", overall_max05_confidences)
    
    # overall_max06_confidences = np.mean(average_max06_append_confidence) if average_max06_append_confidence else 0.0
    # print(f"Maximum_06_average_confidence: ", overall_max06_confidences)
    
    #所有圖大於0.5的匹配線數量
    print(f"Overall count of values greater than 0.5 across all lines: {count_above_05_total}")
    
    # print(f"Overall count of values greater than 0.6 across all lines: {count_above_06_total}")
    
    # 所有圖的匹配線總數量
    print(f"Overall count of all lines: {count_all}")
    
    #confidence大於0.5的比率
    # print(f"Confidence ratio of more than 0.5: ({count_above_05_total}/{count_all}), {count_above_05_total/count_all}")
    
    #將圖片結果轉換成影片
    # image_folder = 'dump_demo_sequence'  #圖片序列的資料夾
    # output_video_path = 'D:/image_experience/SuperGluePretrainedNetwork-master/1123_True.mp4'  # 輸出影片的路徑
    # fps = 20  #影片每秒的幀數
    # images_to_video(image_folder, output_video_path, fps) #呼叫函數將圖片序列轉為影片
    
    cv2.destroyAllWindows()
    vs.cleanup()
