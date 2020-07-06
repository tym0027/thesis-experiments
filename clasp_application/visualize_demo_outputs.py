import numpy as np
import cv2
import os
import glob

import torch

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import codecs, json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--saveDir', default='/data/ALERT-SHARE/may_demo/', help='Location to save frames for visualization...')
parser.add_argument('--input_bins', default='./20191024-cam9/formatted_bins_09.txt', help='Location of bin data...')
parser.add_argument('--input_people', default='./20191024-cam9/cam09exp2_logs_fullv1.txt', help='Location of human data...')
parser.add_argument('--input_step', default='./20191024-cam9/201910-cam9-td.npz', help='Location of step data...')
parser.add_argument('--travel_units', default='', help='Location of travel unit data...')
parser.add_argument('--input_frames', default='/home/rsl/ws/CLASP2-Association/data/cam09exp2/',  help='Location of frame data...')
parser.add_argument('--checkpoint', default='', help='Set if you have associations from previous cameras...')
parser.add_argument('--save_checkpoint', default='.', help='Where to save associations from camera...')
parser.add_argument('--visualize', action='store_true', help='Whether or not you save frames with output...')

args = parser.parse_args()

def json_save(path, data):
    list_data = data
    list_data["amatrix"] = list_data["amatrix"].tolist()
    json.dump(list_data, codecs.open(path + '/association_data.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    data["amatrix"] = np.array(data["amatrix"])

def json_load(path):
    obj = codecs.open(path, 'r', encoding='utf-8').read()
    data = json.loads(obj)
    data["amatrix"] = np.array(data["amatrix"])

    return data

# TINT_COLOR = (0, 255, 0)
TRANSPARENCY = .25  # Degree of transparency, 0-100%
OPACITY = int(255 * TRANSPARENCY)

font = ImageFont.truetype("./arial.ttf", 48) # originally 28
classes = ["P2P", "TRANSFER", "Nothing here..."]
IOU_THRESH = .5


def draw_shaded_rectangle(f, coords, TINT_COLOR):
    img = Image.open(f)
    img = img.convert("RGBA")

    overlay = Image.new('RGBA', img.size, TINT_COLOR+(0,))
    draw_alpha = ImageDraw.Draw(overlay)
    # draw.polygon([(coords[0],coords[1]), (coords[2],coords[3])], (0, 255, 0, 125))
    draw_alpha.rectangle(((coords[0], coords[1]), (coords[2], coords[3])), fill=TINT_COLOR+(OPACITY,))

    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
    img.save(f)


def draw_rectangle(draw, coords, outline, width=8):
    for i in range(int(width/2)):
        draw.rectangle(coords-i, outline=outline, fill=None)
    for i in range(1,int(width/2)):
        draw.rectangle(coords+i, outline=outline, fill=None)


# The niave way of computing bounding boxes...s
def niave(x, y):
    IoU = count = 0
    for indexi in range(0,x.shape[0] - 1):
        for indexj in range(indexi, x.shape[0]):
            if indexj == indexi:
                continue
            elif (x[indexi,0] <= x[indexj,0] <= x[indexi,1]) or (x[indexj,0] <= x[indexi,0] <= x[indexj,1]):
                if (y[indexi,0] <= y[indexj,0] <= y[indexi,1]) or (y[indexj,0] <= y[indexi,0] <=  y[indexj,1]):
                    # print("Rectangle", indexi, "and rectangle", indexj, "intersect...")
                    intersection = (min(x[indexi,1],x[indexj,1]) - max(x[indexi,0],x[indexj,0])) * (min(y[indexi,1],y[indexj,1]) - max(y[indexi,0],y[indexj,0]))
                    union = ((x[indexi,1] - x[indexi,0]) * (y[indexi,1] - y[indexi,0])) + ((x[indexj,1] - x[indexj,0]) * (y[indexj,1] - y[indexj,0])) - intersection
                    area_target = (x[indexj,1] - x[indexj,0]) * (y[indexj,1] - y[indexj,0]) # or??? ((x[indexi,1] - x[indexi,0]) * (y[indexi,1] - y[indexi,0]))
                    
                    # IoU = intersection / union
                    IoU = intersection / area_target
                    count = count + 1;
    return IoU


def associations(f, things, assoc_data):
    if len(things["dets"]) == 0:
        return
    if len(things["bins"]) == 0:
        return
    if len(things["people"]) == 0:
        return

    detections = 0

    bin_data = []
    peep_data = []
    best_iou = iou = 0
    for i in range(0, len(things["dets"])):
        bin_data = []
        best_iou = iou = 0
        
        for j in range(0, len(things["people"])):
            x = np.array([[things["dets"][i][0], things["dets"][i][2]],[things["people"][j][0][0],things["people"][j][0][2]]])
            y = np.array([[things["dets"][i][1], things["dets"][i][3]],[things["people"][j][0][1],things["people"][j][0][3]]])        
            iou = niave(x,y)
            if iou > best_iou:
                best_iou = iou
                peep_data.append((iou, i, j))

        if best_iou > .1:
            bin_iou = 0
            for j in range(0, len(things["bins"])):
                x = np.array([[things["dets"][i][0], things["dets"][i][2]],[things["bins"][j][0][0],things["bins"][j][0][2]]])
                y = np.array([[things["dets"][i][1], things["dets"][i][3]],[things["bins"][j][0][1],things["bins"][j][0][3]]])
                iou = niave(x,y)
                if iou > bin_iou:
                    bin_iou = iou
                    bin_data.append((iou, i, j))
            
            if bin_iou > .1:
                b = bin_data[-1][2]
                
                p = peep_data[-1][2]

                xx = assoc_data["xlabel"].index(things["bins"][b][1])
                yy = assoc_data["ylabel"].index(things["people"][p][1])
                assoc_data["amatrix"][yy,xx] += 1
               
                m = np.amax(assoc_data["amatrix"][:, xx])
                i = np.where(assoc_data["amatrix"][:, xx] == np.amax(assoc_data["amatrix"][:,xx]))[0]
               
                if type(i) != type(int(1)):
                    i = i[0]

                # NOTE: When other cameras start creating 'new' people from switch events,
                # you can break the traveling units feature because the
                # new IDs are not in the original travling units data...
                try:
                    _tu1 = travel_units[assoc_data["ylabel"][int(yy)]]
                except:
                    _tu1 = "NONE"
                try:
                    _tu2 = travel_units[assoc_data["ylabel"][int(i)]]
                except:
                    _tu2 = "NONE"

                
                if "TSO" in assoc_data["ylabel"][int(yy)]:
                    TINT_COLOR = (0, 255, 0)

                    bb = np.asarray(things["bins"][b][0])

                    if args.visualize:
                        draw_shaded_rectangle(f, bb, TINT_COLOR)

                    bb = np.asarray(things["people"][p][0])

                    if args.visualize:
                        draw_shaded_rectangle(f, bb, TINT_COLOR)

                        I = Image.open(f)
                        draw = ImageDraw.Draw(I)
                        draw.text((10, 100 * (detections) + 10), 'Officer {} is associating with B{}'.format(assoc_data["ylabel"][int(yy)], assoc_data["xlabel"][int(xx)]), fill="red",font=font)
                        I.save(f)

                    assoc_data["amatrix"][yy,xx] -= 1
                    detections += 1

                elif (m > 10) and int(i) != int(yy) and (_tu1 != _tu2 or _tu1 == "NONE") and m > 10 and ("TSO" not in assoc_data["ylabel"][int(yy)]):
                    TINT_COLOR = (255, 0, 0)

                    bb = np.asarray(things["bins"][b][0])
                    
                    if args.visualize:
                        draw_shaded_rectangle(f, bb, TINT_COLOR)

                    bb = np.asarray(things["people"][p][0])

                    if args.visualize:
                        draw_shaded_rectangle(f, bb, TINT_COLOR)

                        I = Image.open(f)
                        draw = ImageDraw.Draw(I)
                        draw.text((10, 100 * (detections) + 10), '{} is suspicious with B{}'.format(assoc_data["ylabel"][int(yy)], assoc_data["xlabel"][int(xx)]), fill="red",font=font)
                        I.save(f)

                    assoc_data["amatrix"][yy,xx] -= 1 
                    detections += 1

                # elif m > 10:
                elif assoc_data["amatrix"][yy,xx] >= 10: # and ( assoc_data["amatrix"][yy,xx] >= m):
                    TINT_COLOR = (0, 255, 0)
                    bb = np.asarray(things["bins"][b][0])
                    
                    if args.visualize:
                        draw_shaded_rectangle(f, bb, TINT_COLOR)

                    bb = np.asarray(things["people"][p][0])
                    
                    if args.visualize:
                        draw_shaded_rectangle(f, bb, TINT_COLOR)

                        I = Image.open(f)
                        draw = ImageDraw.Draw(I)
                        draw.text((10, 100 * (detections) + 10), 'B{} is associated with {}'.format(assoc_data["xlabel"][int(xx)], assoc_data["ylabel"][int(yy)]), fill="green",font=font)
                        I.save(f)

                    detections += 1

    return

# set up travel units
travel_units = {}
if args.travel_units != '':
    TU = open(args.travel_units, 'r').read().split("\n")
    for t in TU:
        if len(t.split(",")) > 1:
            travel_units[t.split(",")[0]] = t.split(",")[1]
        else:
            travel_units[t.split(",")[0]] = "NONE"
print(travel_units)

# saveDir = "/data/ALERT-SHARE/may_demo/"
inputData = sorted(glob.glob(args.input_frames + "*.jpg"))
dataLength = len(inputData)

### load bins
bins = open(args.input_bins, 'r').read().split("\n")
print(bins[0])

### load people
people = open(args.input_people, 'r').read().split("\n")
print(people[0])

### load step detections
if ".npz" in args.input_step:
    step_detections = np.load(args.input_step, allow_pickle=True)['allResult']
elif ".txt" in args.input_step or ".csv" in args.input_step:
    step_detections = open(args.input_step,'r').read().split("\n")
print(step_detections[0])

# Asasociation Data
assoc_data = {"xlabel" : [-1], "ylabel" : [-1], "amatrix" : np.zeros((1,1))}
if args.checkpoint != '':
    print("loading data from " + args.checkpoint)
    assoc_data = json_load(args.checkpoint)


print("frame length: ", dataLength)
peep_idx = 0
bin_idx = 0
det_idx = 0
for i in range(0, dataLength):
    things = {"bins" : [], "people" : [], "dets" : []}
    I = Image.open(inputData[i]) # .convert("RGBA")
    W, H = I.size
    
    if args.visualize:
        draw = ImageDraw.Draw(I)
    
    try:
        while str(bins[bin_idx].split(",")[0]).zfill(5) in inputData[i].split("/")[-1]:
            bb = np.asarray([float(bins[bin_idx].split(",")[2])*3, float(bins[bin_idx].split(",")[3])*3, float(bins[bin_idx].split(",")[4])*3, float(bins[bin_idx].split(",")[5])*3], dtype=np.float32)
            if args.visualize:
                # draw_rectangle(draw, bins[bin_idx].split(",")[1:5], outline="blue")
                draw_rectangle(draw, bb, outline="blue")
                draw.text((bb[0]+10, bb[1]+10), 'B {}'.format(bins[bin_idx].split(",")[1]), fill="blue",font=font)
            if not bins[bin_idx].split(",")[1].replace(" ", '') in assoc_data["xlabel"]:
                assoc_data["xlabel"].append(bins[bin_idx].split(",")[1].replace(" ",''))
                assoc_data["amatrix"] = np.concatenate((assoc_data["amatrix"], np.zeros((len(assoc_data["ylabel"]),1))), 1)


            things["bins"].append(([float(bins[bin_idx].split(",")[2])*3, float(bins[bin_idx].split(",")[3])*3, float(bins[bin_idx].split(",")[4])*3, float(bins[bin_idx].split(",")[5])*3], bins[bin_idx].split(",")[1].replace(" ", '')))

            bin_idx += 1;
    except IndexError:
        pass

    try:
        # print(str(people[peep_idx].split(",")[0]).zfill(5), inputData[i].split("/")[-1], str(people[peep_idx].split(",")[0]).zfill(5) in inputData[i].split("/")[-1])
        while str(people[peep_idx].split(",")[0]).zfill(5) in inputData[i].split("/")[-1]:
            bb =  np.asarray([float(people[peep_idx].split(",")[2]), float(people[peep_idx].split(",")[3]), float(people[peep_idx].split(",")[4]), float(people[peep_idx].split(",")[5])], dtype=np.float32)
            
            if args.visualize:
                # draw_rectangle(draw, people[peep_idx].split(",")[2:5], outline="lightblue")
                draw_rectangle(draw, bb, outline="blue")
                draw.text((bb[0]+10, bb[1]+10), '{}'.format(people[peep_idx].split(",")[1]), fill="blue",font=font)

            if not people[peep_idx].split(",")[1].replace(" ", '') in assoc_data["ylabel"]:
                assoc_data["ylabel"].append(people[peep_idx].split(",")[1].replace(" ", ''))
                assoc_data["amatrix"] = np.concatenate((assoc_data["amatrix"], np.zeros((1,len(assoc_data["xlabel"])))), 0)

            things["people"].append(([float(people[peep_idx].split(",")[2]), float(people[peep_idx].split(",")[3]), float(people[peep_idx].split(",")[4]), float(people[peep_idx].split(",")[5])], people[peep_idx].split(",")[1].replace(" ", '')))

            peep_idx += 1;
    except IndexError:
        pass

    try:
        while str(step_detections[det_idx].split(",")[1]).zfill(5) in inputData[i].split("/")[-1]:
            datum = step_detections[det_idx].split(",")
            bb = np.asarray([float(datum[2])*W, float(datum[3])*H, float(datum[4])*W, float(datum[5])*H], dtype=np.float32);
            action = datum[6]
            if int(action) - 1 == 2: # background
                pass
            elif float(datum[7]) > IOU_THRESH: # NOTE: THIS IS A CONFIDENCE THRESHOLD NOT A IOU THRESHOLD
                things["dets"].append([float(datum[2])*W, float(datum[3])*H, float(datum[4])*W, float(datum[5])*H])
                
                if args.visualize:
                    draw_rectangle(draw, bb, outline="red")
                    draw.text((bb[0]+10, bb[1]+10), '{}'.format(classes[int(action) - 1]), fill="red",font=font)
            else:
                pass

            det_idx += 1;
    except IndexError:
        pass

    if args.visualize:
        I = I.convert("RGB")
        I.save(args.saveDir + inputData[i].split("/")[-1])

    associations(args.saveDir + inputData[i].split("/")[-1],  things, assoc_data)
    json_save(args.save_checkpoint, assoc_data)
    print("saving to ", args.saveDir + inputData[i].split("/")[-1])
