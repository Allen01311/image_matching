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
import random
import numpy as np
import matplotlib.cm as cm
import torch


from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
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
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

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

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    if opt.eval:
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(opt.input_pairs))

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
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

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    #如果opt.eval啟用(特徵匹配點>=38條)，則將評估結果保存到相同的 output_dir 目錄中
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_dir))
    #如果opt.viz啟用，則將可視化圖像保存到相同的 output_dir 目錄中
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)  #計時器
    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2] #從pair中的前兩個元素提取字符串(圖像名稱)
        stem0, stem1 = Path(name0).stem, Path(name1).stem   #.stem:提取屬性。在此用於提取兩個圖像文件的主體名稱，用於命名輸出文件
        #matches_path：匹配結果的二進制文件路徑(匹配的關鍵點和相關資訊)
        #eval_path：評估結果的二進制文件路徑(姿態估計誤差、精度和匹配分數等)
        #viz_path：用於可視化匹配的圖像文件路徑(匹配的關鍵點在圖像上的可視化資訊)
        #viz_eval_path：用於可視化評估結果的圖像文件路徑(姿態估計誤差和精度等的可視化資訊)
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
        viz_eval_path = output_dir / \
            '{}_{}_evaluation.{}'.format(stem0, stem1, opt.viz_extension)

        # Handle --cache logic.
        #控制任務的布林變數
        do_match = True #是否執行匹配操作。如果opt.cache被禁用或輸出路徑中的matches_path文件不存在，則設置為True
        do_eval = opt.eval  #是否執行評估功能。如果opt.cache被禁用或輸出路徑中的eval_path文件不存在，並且opt.eval被啟用，則設置為True
        do_viz = opt.viz    #是否執行可視化匹配操作。如果opt.cache被禁用或輸出路徑中的viz_path文件不存在，並且opt.viz被啟用，則設置為 True
        do_viz_eval = opt.eval and opt.viz  #是否執行可視化評估操作。如果opt.cache被禁用或輸出路徑中的 viz_eval_path 文件不存在，並且opt.eval和opt.viz同時被啟用，則設置為 True
        
        #檢查是否啟用opt.cache(緩存功能)
        if opt.cache:
            #檢查是否存在與匹配結果相關的緩存資料夾(matches_path)
            if matches_path.exists():
                try:
                    results = np.load(matches_path) #如果存在將load此資料夾以獲得已經計算的匹配結果
                except:
                    raise IOError('Cannot load matches .npz file: %s' %
                                  matches_path)

                #存在且載入後會提取的資訊
                kpts0, kpts1 = results['keypoints0'], results['keypoints1'] #已經計算的特徵點
                matches, conf = results['matches'], results['match_confidence'] #已經計算的特徵對
                do_match = False    #表示不需要再進行匹配，所以設為False
            
            #啟用評估功能(opt.eval)，且相應的評估緩存文件存在(eval_path)
            if opt.eval and eval_path.exists():
                try:
                    results = np.load(eval_path)    #嘗試載入緩存文件eval_path
                except:
                    raise IOError('Cannot load eval .npz file: %s' % eval_path) 
                err_R, err_t = results['error_R'], results['error_t']#旋轉誤差(err_R)、位移誤差(err_t)
                precision = results['precision']    #已經計算的匹配精確度(precision)
                matching_score = results['matching_score']  #已經計算的匹配得分(matching_score)
                num_correct = results['num_correct']    #已經計算的正確匹配數量(num_correct)
                epi_errs = results['epipolar_errors']   #已經計算的極線誤差(epipolar_errors)
                do_eval = False                     #eval_path緩存已存在，所以設為False
            #可視化匹配判斷式
            if opt.viz and viz_path.exists():
                do_viz = False
            #可視化評估結果判斷式
            if opt.viz and opt.eval and viz_eval_path.exists():
                do_viz_eval = False
            timer.update('load_cache')  #用於追蹤不同部分的代碼執行時間。"load_cache"表示已經載入緩存資訊。

        #如果4個功能都不需要執行(皆為False)則執行以下操作
        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs))) #處理了第i對圖像中的一對(對每一對特徵點都執行一次)
            continue

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:  #檢查pair列表長度(若>=5則可能包含圖像旋轉的資訊)
            rot0, rot1 = int(pair[2]), int(pair[3]) #把pair中索引2、3的元素(圖像旋轉資訊)轉換為整數，並分別存儲在rot0和rot1中
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        #進行預處理
        #調用read_image函數來讀取第一個圖像(name0)
        #name:從input_dir路徑組合而成的(圖像的位置)
        #device:處理圖像的設備(例如，CPU 或 GPU)
        #opt.resize:布爾值，表示是否要調整圖像大小
        #rot:是圖像的旋轉角度
        #opt.resize_float:調整大小時使用的浮點值
        image0, inp0, scales0 = read_image(
            input_dir / name0, device, opt.resize, rot0, opt.resize_float)
        image1, inp1, scales1 = read_image(
            input_dir / name1, device, opt.resize, rot1, opt.resize_float)
        #檢查圖像是否成功讀取
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        timer.update('load_image')  #記錄圖像讀取操作所花費的時間

#-----------------------以下為各功能Function-----------------------

        #do_match = True則表示需要執行圖像匹配操作
        if do_match:
            # Perform the matching.
            #將預處理後的圖像inp0和inp1輸入至matching函數執行圖像匹配操作。匹配的結果存儲在pred中
            pred = matching({'image0': inp0, 'image1': inp1})
            #將pred中的結果轉換為NumPy數組，並將它們轉移到CPU(更容易處理和存儲結果)
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            #從pred中提取特徵點(keypoints)的位置資訊，分別存儲在kpts0和kpts1中
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            #從pred中提取匹配對及匹配信心(confidence)的資訊，分別存儲在matches和conf中
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher') #記錄圖像匹配操作花費的時間

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            #將out_matches中的匹配結果以NumPy的.npz格式保存到指定的文件路徑matches_path中
            np.savez(str(matches_path), **out_matches)

        # Keep the matching keypoints.
        valid = matches > -1    #標記有效匹配(值大於-1表示有效匹配，而-1表示無效匹配)
        #從kpts0和kpts1中提取有效的特徵點位置，分別存儲在mkpts0和mkpts1中(目的:只保留有效的匹配特徵點)
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid] #從conf中提取有效的匹配信心(confidence)值，存儲在mconf中(目的:保留有效匹配的信心值)


        if do_eval: #檢查是否需要執行評估
            # Estimate the pose and compute the pose error.
            #確保pair的數量=38條
            #assert用法:assert condition, error_message
            assert len(pair) == 38, 'Pair does not have ground truth info'
            #相機校準矩陣資訊
            K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            # Scale the intrinsics to resized image.
            #對相機校準矩陣進行縮放，以適應圖像的縮放。scales0、scales1:縮放的因子
            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)

            # Update the intrinsics + extrinsics if EXIF rotation was found.
            if rot0 != 0 or rot1 != 0:
                cam0_T_w = np.eye(4)
                cam1_T_w = T_0to1
                if rot0 != 0:
                    K0 = rotate_intrinsics(K0, image0.shape, rot0)
                    cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                if rot1 != 0:
                    K1 = rotate_intrinsics(K1, image1.shape, rot1)
                    cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                T_0to1 = cam1_T_cam0

            #compute_epipolar_error:計算一組特徵點mkpts0和mkpts1的楕圓線誤差(epipolar error)
            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            correct = epi_errs < 5e-4   #將epipolar誤差與一個閾值比較，以確定哪些特徵點對是一致的
            num_correct = np.sum(correct)   #成功特徵點的配對數
            
            #每個特徵點進行配對的成功率
            precision = np.mean(correct) if len(correct) > 0 else 0
            
            #匹配分數。所有特徵點成功配對的成功率
            matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0  

            #估計姿態的相關閾值參數
            thresh = 1.  # In pixels relative to resized image size.
            #使用特徵點mkpts0和mkpts1、相機內部參數K0和K1、閾值thresh來估計相對姿態
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
            if ret is None: #若ret is Non，則表示估計姿態失敗
                err_t, err_R = np.inf, np.inf   #設為正無窮表示失敗
            else:           #若估計姿態成功
                R, t, inliers = ret #則回傳資訊:R旋轉、t平移、inliers內點
                err_t, err_R = compute_pose_error(T_0to1, R, t) #並分別計算平移(err_t)和旋轉(err_R)的誤差

            # Write the evaluation results to disk.
            #將估計的誤差、精度和匹配分數、epipolar誤差保存為NumPy.npz文件，以利於後續分析
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        'epipolar_errors': epi_errs}
            np.savez(str(eval_path), **out_eval)
            timer.update('eval')    #記錄姿態估計執行的時間


        if do_viz:#檢查是否需要進行可視化
            # Visualize the matches.
            color = cm.jet(mconf)   #特徵點匹配顏色，不同可信度值對應到不同的顏色
            #文字資訊:"SuperGlue"字樣、參與匹配的特徵點數量、是否存在圖像旋轉等等
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]   
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            #顯示較小的文字:閾值、圖像對的名稱
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            #可視化函式
            #image0、image1：要匹配的兩個圖像
            #kpts0、kpts1：圖像的特徵點
            #mkpts0、mkpts1：成功匹配的特徵點
            #color：特徵點匹配的顏色，根據可信度而定
            #text：要在圖像上顯示的主要資訊
            #viz_path：保存可視化圖像的路徑
            #opt.show_keypoints：一個選項，指示是否在可視化中顯示特徵點
            #opt.fast_viz：一個選項，指示是否使用快速可視化模式
            #opt.opencv_display：一個選項，指示是否使用OpenCV顯示圖像
            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')   #記錄可視化過程花費的時間


        if do_viz_eval: #檢查是否需要進行相對姿態估計結果的可視化
            # Visualize the evaluation results for the image pair.
            #epi_errs:(視差誤差)的值，透過線性映射將誤差值轉換為顏色
            #誤差值越小，顏色越接近1，誤差值越大，顏色越接近0
            color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)  #算值
            color = error_colormap(1 - color)   #值轉換成顏色
            deg, delta = ' deg', 'Delta '   #角度單位
            if not opt.fast_viz:
                deg, delta = '°', '$\\Delta$'
            e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)  #位移(平移)的誤差
            e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)  #旋轉的誤差
            #"SuperGlue"字樣、位移誤差、旋轉誤差、匹配的內點數量
            text = [
                'SuperGlue',
                '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info (only works with --fast_viz).
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]
            #生成相對姿態估計結果的可視化圖像
            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0,
                mkpts1, color, text, viz_eval_path,
                opt.show_keypoints, opt.fast_viz,
                opt.opencv_display, 'Relative Pose', small_text)

            timer.update('viz_eval')    #可視化姿態估計花費的時間

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs))) #當前所有行為花費時間


    if opt.eval:    #檢查是否執行評估操作
        # Collate the results into a final table and print to terminal.
        pose_errors = []        #姿勢誤差
        precisions = []         #精度
        matching_scores = []    #匹配分數
        for pair in pairs:      #迭代處理每個圖像
            name0, name1 = pair[:2] #圖像文件名
            stem0, stem1 = Path(name0).stem, Path(name1).stem   #圖像基本名稱
            #保存路徑(保存在.npz文件中)
            eval_path = output_dir / \
                '{}_{}_evaluation.npz'.format(stem0, stem1)
            results = np.load(eval_path)    #載入評估結果(姿勢誤差、精度、匹配分數)
            #計算姿勢誤差，取error_t(平移誤差)和error_R(旋轉誤差)之中的較大值
            #並用於計算Area Under the Curve(AUC)
            pose_error = np.maximum(results['error_t'], results['error_R'])
            pose_errors.append(pose_error)  #將上述較大值之誤差姿勢添加至pose_errors
            precisions.append(results['precision']) #將精度添加到 precisions
            matching_scores.append(results['matching_score'])  #將匹配分數添加到 matching_scores
        thresholds = [5, 10, 20]    #計算AUC的閾值
        aucs = pose_auc(pose_errors, thresholds)    #將姿勢誤差和閾值帶入pose_auc函數計算AUC
        aucs = [100.*yy for yy in aucs]             #將AUC值轉換為百分比
        prec = 100.*np.mean(precisions)             #計算精度的平均值，並轉換為百分比
        ms = 100.*np.mean(matching_scores)          #計算匹配分數的平均值，並轉換為百分比
        #print出評估結果至terminal
        print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], prec, ms))
