
import darknet as dn
import os
import numpy as np
import collections
import np_box_ops as bo
import winsound
from PIL import Image
import json

#find duplicates function
def list_duplicates(n):
     counter=collections.Counter(n) 
     dups=[i for i in counter if counter[i]!=1] 
     result={}
     for item in dups:
             result[item]=[i for i,j in enumerate(n) if j==item] 
     return result

#yolo configs, get the cfg files and weight files from the links in the repo 
yolo_v2_cfg = "./cfg/yolo.cfg"
yolo_v2_weights = "./weights/yolo.weights"

yolo_v3_cfg = "./cfg/yolov3-spp.cfg"
yolo_v3_weights = "./weights/yolov3-spp.weights"

yolo_v4_cfg = "./cfg/yolov4.cfg"
yolo_v4_weights = "./weights/yolov4.weights"

#put the paths of the various datasets for evaluation here:
#fpath_clean is the clean dataset, before applying the patch
#fpath_patch is the attacked dataset, after applying the patch
#fpath_inpaint is the recovered dataset, after running Jedi
fpath_clean = ""
fpath_patch = ""
fpath_inpaint = ""
folder = os.listdir(fpath_clean)

results_c = []
results_p = []
results_i = []
i = 0
f = 0

detected = 0
patch_success = 0
patch_fail = 0
recovered = 0
gt_total = 0
lost = 0

for im in folder:
    img_path_clean = fpath_clean + im
    
    # print('==========================')
    # print('Image: ' + im)
    # print('==========================')
    
    #######
    #clean#
    #######
    
    result_c = dn.performDetect(img_path_clean,
                                   0.5,
                                   yolo_v2_cfg,
                                   yolo_v2_weights,
                                   "./cfg/coco.data",
                                   showImage= False,
                                   makeImageOnly = False,
                                   initOnly= False)
    
    results_c.append(result_c)
    i = i+1
    

    img_c = Image.open(img_path_clean)
     
    #ground ruths go here
    
    
    f=f+1
    
    person_boxes_c = []
    person_scores_c = []
    
    for j in range(len(result_c)):
        if result_c[j][0] == 'person':
            person_scores_c.append(result_c[j][1])
            box = result_c[j][2]
            xmin = int(box[0] - (box[2]/2))
            xmax = int(box[0] + (box[2]/2))
            ymin = int(box[1] - (box[3]/2))
            ymax = int(box[1] + (box[3]/2))
            xmin = max(xmin,0)
            xmax = min(xmax,img_c.size[0])
            ymin = max(ymin,0)
            ymax = min(ymax,img_c.size[1])
            new_box = [xmin,ymin,xmax,ymax]
            person_boxes_c.append(new_box)
            
            
    person_boxes_c = np.array(person_boxes_c)
    person_scores_c = np.array(person_scores_c)
    
    #calculate ious and assign detections to gts
    if len(person_boxes_c) == 0:
        assignment_c = [-1] * len(frame_gt)
        asn_scores_c = [0] * len(frame_gt)
    else:
        ious_c = bo.iou(person_boxes_c,frame_gt)
        ious_c = np.around(ious_c,4)
        assignment_c = list([-1] * len(frame_gt))
        asn_scores_c = [0] * len(frame_gt)
        for i in range(len(frame_gt)):
            if ious_c.size == 0:
                break
            assign_c = np.where(ious_c[:,i] == max(ious_c[:,i]))
            if max(ious_c[:,i]) > 0.5:
                assignment_c[i] = assign_c[0][0]
                asn_scores_c[i] = person_scores_c[assign_c[0][0]]
            else:
                pass
            
    #remove duplicate assignments
    dupes_c = list_duplicates(assignment_c)
    for i in dupes_c:
        if i != -1:
            best_iou_pos_c = np.where(ious_c[i,dupes_c.get(i)] == max(ious_c[i,dupes_c.get(i)]))
            best_iou_c = dupes_c[i][best_iou_pos_c[0][0]]
            for j in dupes_c[i]:
                if j != best_iou_c:
                    assignment_c[j] = -1
                    asn_scores_c[j] = 0
                    
    gt = len(frame_gt)
    tp_c = gt - collections.Counter(assignment_c)[-1]
           
    gt_total = gt_total + gt
    detected = detected + tp_c
    
    #######
    #patch#
    #######
    
    img_path_patch = fpath_patch + im
        
    result_p = dn.performDetect(img_path_patch,
                                0.5,
                                yolo_v2_cfg,
                                yolo_v2_weights,
                                "./cfg/coco.data",
                                showImage= False,
                                makeImageOnly = False,
                                initOnly= False)
    
    results_p.append(result_p)
    
    img_p = Image.open(img_path_patch)
    
    person_boxes_p = []
    person_scores_p = []
    
    for j in range(len(result_p)):
        if result_p[j][0] == 'person':
            person_scores_p.append(result_p[j][1])
            box = result_p[j][2]
            xmin = int(box[0] - (box[2]/2))
            xmax = int(box[0] + (box[2]/2))
            ymin = int(box[1] - (box[3]/2))
            ymax = int(box[1] + (box[3]/2))
            xmin = max(xmin,0)
            xmax = min(xmax,img_p.size[0])
            ymin = max(ymin,0)
            ymax = min(ymax,img_p.size[1])
            new_box = [xmin,ymin,xmax,ymax]
            person_boxes_p.append(new_box)
            
            
    person_boxes_p = np.array(person_boxes_p)
    person_scores_p = np.array(person_scores_p)
    
    #calculate ious and assign detections to gts
    if len(person_boxes_p) == 0:
        assignment_p = [-1] * len(frame_gt)
        asn_scores_p = [0] * len(frame_gt)
    else:
        ious_p = bo.iou(person_boxes_p,frame_gt)
        ious_p = np.around(ious_p,4)
        assignment_p = list([-1] * len(frame_gt))
        asn_scores_p = [0] * len(frame_gt)
        for i in range(len(frame_gt)):
            if ious_p.size == 0:
                break
            assign_p = np.where(ious_p[:,i] == max(ious_p[:,i]))
            if max(ious_p[:,i]) > 0.5:
                assignment_p[i] = assign_p[0][0]
                asn_scores_p[i] = person_scores_p[assign_p[0][0]]
            else:
                pass
            
    #remove duplicate assignments
    dupes_p = list_duplicates(assignment_p)
    for i in dupes_p:
        if i != -1:
            best_iou_pos_p = np.where(ious_p[i,dupes_p.get(i)] == max(ious_p[i,dupes_p.get(i)]))
            best_iou_p = dupes_p[i][best_iou_pos_p[0][0]]
            for j in dupes_p[i]:
                if j != best_iou_p:
                    assignment_p[j] = -1
                    asn_scores_p[j] = 0
                    
    patch_successes = [0] * len(frame_gt)
    for det in range(len(assignment_p)):
        #debug
        #print("#"+str(det)+": clean_det?:"+str(assignment_c[det])+" patch_det?:"+str(assignment_p[det]))
        if (assignment_c[det] != -1 and assignment_p[det] == -1):
            #print("#"+str(det)+" Successful patch")
            patch_successes[det] = 1
            
            
            
    p_suc = collections.Counter(patch_successes)[1]
    p_fail = tp_c - p_suc
    
    patch_success = patch_success + p_suc
    patch_fail = patch_fail + p_fail
    
    ############
    #Inpainting#
    ############
    
    img_path_inpaint = fpath_inpaint + im
        
    result_i = dn.performDetect(img_path_inpaint,
                                0.5,
                                yolo_v2_cfg,
                                yolo_v2_weights,
                                "./cfg/coco.data",
                                showImage= False,
                                makeImageOnly = False,
                                initOnly= False)
    
    results_i.append(result_i)
    
    img_i = Image.open(img_path_inpaint)
    
    person_boxes_i = []
    person_scores_i = []
    
    for j in range(len(result_i)):
        if result_i[j][0] == 'person':
            person_scores_i.append(result_i[j][1])
            box = result_i[j][2]
            xmin = int(box[0] - (box[2]/2))
            xmax = int(box[0] + (box[2]/2))
            ymin = int(box[1] - (box[3]/2))
            ymax = int(box[1] + (box[3]/2))
            xmin = max(xmin,0)
            xmax = min(xmax,img_i.size[0])
            ymin = max(ymin,0)
            ymax = min(ymax,img_i.size[1])
            new_box = [xmin,ymin,xmax,ymax]
            person_boxes_i.append(new_box)
            
            
    person_boxes_i = np.array(person_boxes_i)
    person_scores_i = np.array(person_scores_i)
    
    #calculate ious and assign detections to gts
    if len(person_boxes_i) == 0:
        assignment_i = [-1] * len(frame_gt)
        asn_scores_i = [0] * len(frame_gt)
    else:
        ious_i = bo.iou(person_boxes_i,frame_gt)
        ious_i = np.around(ious_i,4)
        assignment_i = list([-1] * len(frame_gt))
        asn_scores_i = [0] * len(frame_gt)
        for i in range(len(frame_gt)):
            if ious_i.size == 0:
                break
            assign_i = np.where(ious_i[:,i] == max(ious_i[:,i]))
            if max(ious_i[:,i]) > 0.5:
                assignment_i[i] = assign_i[0][0]
                asn_scores_i[i] = person_scores_i[assign_i[0][0]]
            else:
                pass
            
    #remove duplicate assignments
    dupes_i = list_duplicates(assignment_i)
    for i in dupes_i:
        if i != -1:
            best_iou_pos_i = np.where(ious_i[i,dupes_i.get(i)] == max(ious_i[i,dupes_i.get(i)]))
            best_iou_i = dupes_i[i][best_iou_pos_i[0][0]]
            for j in dupes_i[i]:
                if j != best_iou_i:
                    assignment_i[j] = -1
                    asn_scores_i[j] = 0
                    
    recovery_successes = [0] * len(frame_gt)
    for det in range(len(assignment_i)):
        #debug
        #print("#"+str(det)+" c?:"+str(assignment_c[det])+" p?:"+str(assignment_p[det])+" i?:"+str(assignment_i[det]))
        if (patch_successes[det] == 1 and assignment_i[det] != -1):
            #print("#"+str(det)+" Recovery Success")
            recovery_successes[det] = 1
            
    losses = [0] * len(frame_gt)
    for det in range(len(assignment_i)):
        #debug
        #print("#"+str(det)+" c?:"+str(assignment_c[det])+" p?:"+str(assignment_p[det])+" i?:"+str(assignment_i[det]))
        if (assignment_c[det] != -1 and patch_successes[det] == 0 and assignment_i[det] == -1):
            #print("#"+str(det)+" Recovery Success")
            losses[det] = 1
            
    recov = collections.Counter(recovery_successes)[1]
    ls = collections.Counter(losses)[1]
    
    recovered = recovered + recov
    lost = lost + ls
    
    print()
    print("Image " + str(f)) 
    
    if (detected > 0 and patch_success > 0 and patch_fail > 0):
        det_pct = round((detected/gt_total) * 10000) / 100
        ptc_pct = round((patch_success/detected) * 10000) / 100
        rcv_pct = round((recovered/patch_success) * 10000) / 100
        los_pct = round((lost/patch_fail) * 10000) / 100
        
        print("Correct detections: " + str(detected) + " / " + str(gt_total) + "  (+" + str(tp_c) +") (" +str(det_pct)+")" )
        print("Successful patches: " + str(patch_success) + " / " + str(detected) + "  (+" + str(p_suc) +") (" +str(ptc_pct)+")" )
        print("Successful recoveries: " + str(recovered) + " / " + str(patch_success) + "  (+" + str(recov) +") (" +str(rcv_pct)+")" )
        print("Lost Detections: " + str(lost) + " / " + str(patch_fail) + "  (+" + str(ls) +") (" +str(los_pct)+")" )
    
    else: 
        print("Correct detections: " + str(detected) + " / " + str(gt_total) + "  (+" + str(tp_c) +")")
        print("Successful patches: " + str(patch_success) + " / " + str(detected) + "  (+" + str(p_suc) +")" )
        print("Successful recoveries: " + str(recovered) + " / " + str(patch_success) + "  (+" + str(recov) +")" )
        print("Lost Detections: " + str(lost) + " / " + str(patch_fail) + "  (+" + str(ls) +")" )
