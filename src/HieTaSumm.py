# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
from datetime import timedelta
import numpy as np
import cv2 as cv
from sklearn.metrics.pairwise import cosine_similarity
import torch
from PIL import Image
import torchvision.transforms as transforms
import json
import warnings
import networkx as nx
import imageio
# %matplotlib inline
import matplotlib.pyplot as plt
import higra as hg
import os
import sys
from scipy import spatial

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from IPython.display import HTML, display
from glob import glob
import time

try:
    from utils import * # imshow, locate_resource, get_sed_model_file
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())

warnings.filterwarnings('ignore')
# %matplotlib inline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

def create_folder(folder_name):
    """ Create folder if there is not
    Args:
        folder_name: String, folder name
    Returns:
        None
    """
    if not os.path.isdir(f"../models/{folder_name}"):
        os.makedirs(f"../models/{folder_name}")
        print(f"Folder '../models/{folder_name}' created")

def load_model(model_name, include_top=True):
    """ Load pre-trained Keras model
    Args:
        model_name: String, name of model to load
        include_top: String, the model is buildt with 'feature learning block' + 'classification block'
    Returns:
        model: Keras model instance
    """
    if selected_model in available_models:
        # Load a Keras instance
        try:
            if model_name == 'vgg16':
                model = VGG16(weights='imagenet', include_top=include_top)
            elif model_name == 'resnet50':
                model = ResNet50(weights='imagenet', include_top=include_top)
            print(f">> '{model.name}' model successfully loaded!")
        except:
            print(f">> Error while loading model '{selected_model}'")
    
    # Wrong selected model
    else:
        print(f">> Error: there is no '{selected_model}' in {available_models}")
    
    return model

available_models = ['vgg16', 'resnet50']
selected_model = 'resnet50'
model = load_model(selected_model, include_top=True)

def get_img_size_model(model):
    """Returns image size for image processing to be used in the model
    Args:
        model: Keras model instance 
    Returns:
        img_size_model: Tuple of integers, image size
    """
    model_name = model.name
    if model_name == "vgg16":
        img_size_model = (224, 224)
    elif model_name == "resnet50":
        img_size_model = (224, 224)
    else:
        img_size_model = (224, 224)
        print("Warning: model name unknown. Default image size: {}".format(img_size_model))
        
    return img_size_model

def get_layername_feature_extraction(model):
    """ Return the name of last layer for feature extraction   
    Args:
        model: Keras model instance
    Returns:
        layername_feature_extraction: String, name of the layer for feature extraction
    """
    model_name = model.name
    if model_name == "vgg16":
        layername_feature_extraction = 'fc2'
    elif model_name == "resnet50":
        layername_feature_extraction = 'predictions'
    else:
        layername_feature_extraction = ''
        print("Warning: model name unknown. Default layername: '{}'".format(layername_feature_extraction))
    
    return layername_feature_extraction

def get_layers_list(model):
    """Get a list of layers from a model
    Args:
        model: Keras model instance
    Returns:
        layers_list: List of string of layername
    """
    layers_list = [model.layers[i].name for i in range(len(model.layers))]
        
    return layers_list

def image_processing(img_array):
    """ Preprocess image to be used in a keras model instance
    Args:
        img_array: Numpy array of an image which will be predicte
    Returns:
        processed_img = Numpy array which represents the processed image
    """    
    # Expand the shape
    img = np.expand_dims(img_array, axis=0)

    # Convert image from RGB to BGR (each color channel is zero-centered with respect to the ImageNet dataset, without scaling)
    processed_img = preprocess_input(img)
    
    return processed_img

def get_feature_vector(model, img_path):
    """ Get a feature vector extraction from an image by using a keras model instance
    Args:
        model: Keras model instance used to do the classification.
        img_path: String to the image path which will be predicted
    Returns:
        feature_vect: List of visual feature from the input image
    """
    
    # Creation of a new keras model instance without the last layer
    layername_feature_extraction = get_layername_feature_extraction(model)
    model_feature_vect = Model(inputs=model.input, outputs=model.get_layer(layername_feature_extraction).output)
    
    # Image processing
    img_size_model = get_img_size_model(model)
    img = tf.keras.utils.load_img(img_path, target_size=img_size_model)
    img_arr = np.array(img)
    img_ = image_processing(img_arr)
    
    # Visual feature extraction
    feature_vect = model_feature_vect.predict(img_)
    
    return feature_vect

def calculate_similarity(vector1, vector2):
    """Compute similarities between two images using 'cosine similarities'
    Args:
        vector1: Numpy vector to represent feature extracted vector from image 1
        vector2: Numpy vector to represent feature extracted vector from image 1
    Returns:
        sim_cos: Float to describe the similarity between both images
    """
    sim_cos = np.linalg.norm(vector1-vector2, 1) * 100 / len(vector2)
    # sim_cos = 100 * cosine_similarity(vector1, vector2, dense_output=True)[0][0]#1-spatial.distance.cosine(vector1, vector2)
    return sim_cos

def compute_similarity_img(model, img_path_1, img_path_2, fea_vec_img1, fea_vec_img2): # img_path_1, img_path_2):
    """ Return a cosine similarity between both images and display them in HTML
    Args:
        model: Keras model instance used to do the feature extraction
        img_path_1: String to the image 1 path
        img_path_2: String to the image 2 path
    Returns:
        sim_cos: Float to describe the similarity between both images
    """
    filename1 = os.path.basename(img_path_1).split(".")[0]
    filename2 = os.path.basename(img_path_2).split(".")[0]

    # Compute cosine similarity
    sim_cos = calculate_similarity(fea_vec_img1, fea_vec_img2)
    
    # Read images
    img_size_model = get_img_size_model(model)
    im1 = cv2.resize(cv2.imread(img_path_1), dsize=img_size_model, interpolation = cv2.INTER_AREA)
    im2 = cv2.resize(cv2.imread(img_path_2), dsize=img_size_model, interpolation = cv2.INTER_AREA)
    
    # Concatenate images horizontally
    im12 = cv2.hconcat([im1, im2])
    
    # Save concatenated image
    dst_dir_cos_sim = "../report/cos_sim"
    create_folder(dst_dir_cos_sim)
    dst_dir = f"{dst_dir_cos_sim}/{model.name}"
    create_folder(dst_dir)
    
    new_filename = f"{filename1}_{filename2}"
    cv2.imwrite(f"{dst_dir}/{new_filename}.jpg", im12)
    
    return sim_cos

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return (result + ".00") #.replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return "{}.{}".format(result, ms) #:02)

def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = int(cap.get(cv.CAP_PROP_FRAME_COUNT) / cap.get(cv.CAP_PROP_FPS))
    # use np.arange() to make floating-point steps

    for i in np.arange(0, clip_duration, saving_fps):
        s.append(i)
    return s

def frame_extractor(video_file, rate):
    # i.e if video of duration 30 seconds, saves 0.5 frame each second = 60 frames saved in total
    SAVING_FRAMES_PER_SECOND = 1/rate
    filename, _ = os.path.splitext(video_file)
    filename = "./frames/" + filename.split('/')[-1]
    print("-----------------------")
    print(filename)
    # make a folder by the name of the video file
    if not os.path.isdir(filename):
        os.mkdir(filename)
        # read the video file
        cap = cv.VideoCapture(video_file)
        # get the FPS of the video
        fps = cap.get(cv.CAP_PROP_FPS)
        # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
        saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
        # get the list of duration spots to save
        saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
        # start the loop
        count = 0
        frame_number = 1
        while True:
            is_read, frame = cap.read()
            if not is_read:
                # break out of the loop if there are no frames to read
                break
            # get the duration by dividing the frame count by the FPS
            frame_duration = count / fps
            try:
                # get the earliest duration to save
                closest_duration = saving_frames_durations[0]
            except IndexError:
                # the list is empty, all duration frames were saved
                break
            if frame_duration >= closest_duration:
                # if closest duration is less than or equals the frame duration,
                number = str(frame_number).zfill(6)
                frame_number += 1
                cv.imwrite(os.path.join(filename, "{}.jpg".format(number)), frame) # frame_duration_formatted)), frame)
                # drop the duration spot from the list, since this duration spot is already saved
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            # increment the frame count
            count += 1

def ext_resnet(file_dir, net, utils):
  uris = [
      file_dir
  ]

  batch = torch.cat(
      [utils.prepare_input_from_uri(uri) for uri in uris]
  ).to(device)
  return batch
  #with torch.no_grad():
   #   output = torch.nn.functional.softmax(net(batch), dim=1)
  #return output

def ext_features(file_dir, model):
  input_image = Image.open(file_dir)
  preprocess = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
      output = model(input_batch)
  return output[0]
  # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
  # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
  #probabilities = torch.nn.functional.softmax(output, dim=1)
  #return probabilities#.numpy()#.reshape(-1, 1)

def cos_similarity(frame1, frame2): #images to compute a similarity with cosine_similarity metric
  cos_sim =100 *cosine_similarity(frame1, frame2, dense_output=True)[0][0]
  if cos_sim < 20:
    cos_sim = 20

  return float(cos_sim)

def spatialSim(frame1, frame2):
  similarity = 100 * (-1 * (spatial.distance.cosine(frame1, frame2) - 1))
  if similarity < 20:
    similarity = 20
  return similarity


def calc_init(i, delta_t, frame_len):
  if((i < delta_t) or delta_t < 0):
    return 0
  elif((i + delta_t) > frame_len):
    return i
  else:
    return i - delta_t

def calc_end(i, delta_t, frame_len):
  if(((i + delta_t) > frame_len) or delta_t < 0):
    return frame_len
  else:
    return i + delta_t

def saveData(file, v1, v2, weight):
  f = open(file, "a")
  if(v2==' '):
    data = "{}\n".format(v1)
  elif(weight == ".jpg"):
    data = "{}{}\n".format(v1, weight)
  else:
    data = "{}, {}, {}\n".format(v1, v2, weight)
  f.write(data)
  f.close()

def plotGraph(PG, not_weighted):
  elarge = [(u, v) for (u, v, d) in PG.edges(data=True)]
  pos = nx.spring_layout(PG, seed=7)  # positions for all nodes - seed for reproducibility
  nx.draw_networkx_edges(
      PG, pos, edgelist = elarge, width=1, alpha=1
  )

  nx.draw_networkx_labels(PG, pos)

  # nodes
  nx.draw_networkx_nodes(PG, pos)
  if(not_weighted):
    ax = plt.gca()
    plt.show()
  
  else:
    # edge weight labels
    edge_labels = nx.get_edge_attributes(PG, "weight")
    nx.draw_networkx_edge_labels(PG, pos, edge_labels)
    ax = plt.gca()
    plt.show()

def gen_mst(input_graph_file, input_mst):
  VG = readGraphFile(input_graph_file, ' ', cut_graph = False, cutNumber = 0)
  T = nx.minimum_spanning_tree(VG)
  for h in sorted(T.edges(data=True)):
    saveData(input_mst, h[0], h[1], h[2]["weight"])
  return T

def computeHierarchy(input_g, isbinary, input_higra):
  leaf_list = []
  graph = hg.UndirectedGraph()       #convert the scikit image rag to higra unidrect graph
  graph.add_vertices(max(input_g._node)+1)   #creating the nodes (scikit image RAG starts from 1)
  edge_list = list(input_g.edges())                   #ScikitRAG edges
  size_threshold = 20

  for i in range (len(edge_list)):
      graph.add_edge(edge_list[i][0], edge_list[i][1]) #Adding the nodes to higra graph
  edge_weights = np.empty(shape=len(edge_list))
  sources, targets = graph.edge_list()

  for i in range (len(sources)):    
    edge_weights[i] = int(input_g.adj[sources[i]][targets[i]]["weight"])
  nb_tree, nb_altitudes = hg.watershed_hierarchy_by_area(graph, edge_weights)
  
  if(isbinary):
    tree, node_map = hg.tree_2_binary_tree(nb_tree)
    altitudes = nb_altitudes[node_map]
  else:
    tree = nb_tree
    altitudes = nb_altitudes


  for n in tree.leaves_to_root_iterator():
    leaf = -2 # It's cod is used for the node that is not a leaf
    if(tree.is_leaf(n)):
      leaf = -1 # It's cod is used for the node that is a leaf
      leaf_list.append(n)
    saveData(input_higra, n, tree.parent(n), leaf)
  return(leaf_list)

def selectKeyFrame(graph, video_file, leaflist):
  kf_file = video_file + 'keyframe/'
  key_frame = video_file + 'keyframe.txt'
  S = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
  #KF_list = []
  for c in range(len(S)):
    central_node = len(S[c].nodes)
    comp_leaf_list = []
    for i in range(central_node):
      if(list(S[c])[i] in leaflist):
        comp_leaf_list.append(list(S[c])[i])
    cn = int(len(comp_leaf_list)/2)
    kf = str(comp_leaf_list[cn]).zfill(6)

    if not os.path.isdir(kf_file):
      os.mkdir(kf_file)
    
    os.system('cp ' + video_file + kf + '.jpg ' + kf_file + kf + '.jpg')

    saveData(key_frame, kf, '  ', '.jpg')

def readGraphFile(file, cut_graph_file, cut_graph, cutNumber):
  RG = nx.Graph()
  with open(file) as f:
    lines = f.readlines()

  cut = len(lines)
  if(cut_graph):
    cut -= (cutNumber +1)
  cut_list = []
  for line in lines:
    v1 =int(line.split(", ")[0]) # node 1
    v2 =int(line.split(", ")[1]) # node 2
    w = float(line.split(", ")[2]) # weight

    if(cut >= 0):
      RG.add_edge(v1, v2, weight = w) # include two node and your weight
      cut -=1
      if(cut_graph):
        saveData(cut_graph_file, v1, v2, w)

    else:
      if(w == -1):
        v1 =int(line.split(", ")[0]) # node 1
        v2 =int(line.split(", ")[1]) # node 2
        if(v1 <= v2):
          if(not (v2 in cut_list)):
            RG.add_node(v1)
            saveData(cut_graph_file, v1, ' ', ' ')
            cut_list.append(v2)
            cut -=1
        else:
          if(not (v1 in cut_list)):
            RG.add_node(v2)
            saveData(cut_graph_file, v2, ' ', ' ')
            cut_list.append(v1)
            cut -=1
  return RG

def features(video_file):
  model = load_model(selected_model, include_top=True)
  features_list = []
  if(os.path.exists(video_file)):
    frame_list = os.listdir(video_file)
    frame_list.sort() # to garanted the time order
    #model = newLoadNet() # 
    features_list = [    # Compute feature vector extracted
                    get_feature_vector(model, video_file + frames) for frames in frame_list #List of frame features
                    ]
  return features_list

def bestCutNumber(video_file, delta_t):

  if(os.path.exists(video_file)):
    frame_list = os.listdir(video_file)
    frame_list.sort() # to garanted the time order
    features_list = [rgbSim(video_file + frames) for frames in frame_list]
    weight_list = []
    feat_list_len = len(features_list)

    for vertex1 in range(feat_list_len):
      for vertex2 in range(calc_init(vertex1, delta_t, feat_list_len),
                                   calc_end(vertex1, delta_t, feat_list_len)):
        w = spatialSim(features_list[vertex1], features_list[vertex2])
        weight_list.append(w)
    
    print(video_file)
    print(round(np.std(weight_list)) - 1)
        
    return(round(np.std(weight_list)) - 1)

def rgbSim(frame_dir):
    frame = Image.open(frame_dir)
    # make sure images have same dimensions, use .resize to scale image 2 to match image 1 dimensions
    # i am also reducing the shape by half just to save some processing power
    frame_reshape = frame.resize((round(frame.size[0]*0.5), round(frame.size[1]*0.5)))
    # convert the images to (R,G,B) arrays
    frame_array = np.array(frame_reshape)
    # flatten the arrays so they are 1 dimensional vectors
    frame_array = frame_array.flatten()
    # divide the arrays by 255, the maximum RGB value to make sure every value is on a 0-1 scale
    frame_array = frame_array/255
    return frame_array

def cutGraph(file, cut_graph_file, cutNumber):
  RG = nx.Graph()
  cut_list = []

  with open(file) as f:
    lines = f.readlines()
  line = len(lines)-1
  cut_list.append(int(lines[line].split(", ")[0]))

  while(line > 0):
    cut = lines[line]
    v1 =int(cut.split(", ")[0]) # node 1
    v2 =int(cut.split(", ")[1]) # node 2
    w = float(cut.split(", ")[2]) # weight
    
    if (v2 in cut_list):
      if cutNumber >0:
        cutNumber -= 1
        cut_list.append(v1)

      else:
        saveData(cut_graph_file, v1, v2, w)
        RG.add_node(v1)
    else:
      saveData(cut_graph_file, v1, v2, w)
      RG.add_edge(v1, v2, weight = w)
    line -=1

  return RG

def hierarchical_summarization(video_file, rate, time):
  input_graph_file = video_file + 'graph.txt'
  input_mst = video_file + 'mst.txt'
  input_higra = video_file + 'higra.txt'
  cut_graph_file = video_file + 'cut_graph.txt'
  delta_t = rate * time # how many frame to get for the time, for example with time equal to 4 is equivalant to  4 seconds of video and for rate equal to 2 is equal to 2 frames for each second, in this case the firt cut is equal to 8 frames

  if(not os.path.exists(input_graph_file)):
    cut_number = bestCutNumber(video_file, delta_t)
    features_list = features(video_file)
    loadFrames(video_file, delta_t, input_graph_file, features_list) # Load the frame list and create a graph for the video
    print(cut_number)
    tree = gen_mst(input_graph_file, input_mst) # generate the minimum spanning tree
    isbinary = True # To compute a binary hierarchy
    leaflist = computeHierarchy(tree, isbinary, input_higra) # Create the hierarchy based on the minimum spanning tree and return the leaves of the new hierarchy
    cuted_graph = cutGraph(input_higra, cut_graph_file, cutNumber = cut_number) # Create a new graph based on the hierarchy and the level cut
    selectKeyFrame(cuted_graph, video_file, leaflist) # With the cuted graph, create a keyframe to represent each component or segment of video

def bipartiteGraph(frames_dir, gt_file, keyframe_file):  # , output_graph_file, net, utils):
    BPT = nx.Graph()
    ft_gt_list = []
    ft_kf_list = []
    BTM_dict = {}

    if (os.path.exists(gt_file)):
        gt_list = os.listdir(gt_file)
        gt_list.sort()
        ft_gt_list = [
                      get_feature_vector(model, gt_file + frames) for frames in gt_list #List of frame features
                      ]

    if (os.path.exists(keyframe_file)):
        with open(keyframe_file) as f:
            kf_list = f.readlines()
        ft_kf_list = [
                      get_feature_vector(model, frames_dir + frames.split("\\n")[0][:-1]) for frames in kf_list #List of frame features
                      ]

        ft_kf_list_len = len(ft_kf_list)
        ft_gt_list_len = len(ft_gt_list)
        weight_list = []

        for vertex1 in range(ft_kf_list_len):
            for vertex2 in range(ft_gt_list_len):
                w = spatialSim(ft_kf_list[vertex1], ft_gt_list[vertex2])
                weight_list.append(w)
                BPT.add_edge(kf_list[vertex1].split("\\n")[0][:-1], gt_list[vertex2], weight=round(w))
        BTM = sorted(nx.max_weight_matching(BPT))
        BTM_dict = {(u, v): BPT[u][v]["weight"] for u, v in BTM}
    return BTM_dict, ft_kf_list_len, ft_gt_list_len

def Cusa(frames_dir, gt_file, precision):
    total_videos = 0
    cusa_videos_sum = 0
    median_sum_for_cusa = 0
    total_of_medians = 0

    if os.path.exists(frames_dir) and os.path.exists(gt_file):
        video_list = os.listdir(frames_dir)
        video_list.sort()  # to guarantee order
        gt_list = os.listdir(gt_file)
        gt_list.sort()
        for video in video_list:
            gt_user_list = os.listdir("{}/{}/".format(gt_file, video))
            gt_user_list.sort()
            for gt_user in gt_user_list:
                dic_metrics = bipartiteGraph("{}/{}/".format(frames_dir, video),
                                                  "{}/{}/{}/".format(gt_file, video, gt_user),
                                                  "{}/{}/{}".format(frames_dir, video, "keyframe.txt"))
                dic_cut = {k: v for k, v in dic_metrics.items() if v >= precision}
                median = len(dic_cut)/len(os.listdir("{}/{}/{}/".format(gt_file, video, gt_user)))
                median_sum_for_cusa += median
                total_of_medians += 1
            print("for video " + str(video) + " we got a Cusa value of " + str(median_sum_for_cusa/total_of_medians))
            saveData("{}/{}/cusa_{}.txt".format(frames_dir, video, video), 
                     "for video " + str(video) + " we got a Cusa value of " + str(median_sum_for_cusa/total_of_medians)
                     , '  ', '  ')
            cusa_videos_sum += median_sum_for_cusa/total_of_medians
            total_videos += 1
            median_sum_for_cusa = 0
            total_of_medians = 0
        print("Mean of Cusa value of all videos: " + str(cusa_videos_sum/total_videos))
        saveData("{}/{}/results_{}.txt".format(frames_dir, video, precision), "Mean of Cusa value of all videos: " + str(cusa_videos_sum/total_videos), '  ', '  ')

def Cuse(frames_dir, gt_file, precision):
    total_videos = 0
    cuse_videos_sum = 0
    median_sum_for_cuse= 0
    total_of_medians = 0

    if os.path.exists(frames_dir) and os.path.exists(gt_file):
        video_list = os.listdir(frames_dir)
        video_list.sort()  # to guarantee order
        gt_list = os.listdir(gt_file)
        gt_list.sort()
        for video in video_list:
            gt_user_list = os.listdir("{}/{}/".format(gt_file, video))
            gt_user_list.sort()
            for gt_user in gt_user_list:
                dic_metrics = bipartiteGraph("{}/{}/".format(frames_dir, video),
                                                  "{}/{}/{}/".format(gt_file, video, gt_user),
                                                  "{}/{}/{}".format(frames_dir, video, "keyframe.txt"))
                dic_cut = {k: v for k, v in dic_metrics.items() if v < precision}
                median = len(dic_cut)/len(os.listdir("{}/{}/{}/".format(gt_file, video, gt_user)))
                median_sum_for_cuse += median
                total_of_medians += 1
            print("for video " + str(video) + " we got a Cuse value of " + str(median_sum_for_cuse/total_of_medians))
            saveData("{}/{}/cuse_{}.txt".format(frames_dir, video, video), 
                     "for video " + str(video) + " we got a Cuse value of " + str(median_sum_for_cuse/total_of_medians)
                     , '  ', '  ')
            cuse_videos_sum += median_sum_for_cuse/total_of_medians
            total_videos += 1
            median_sum_for_cuse = 0
            total_of_medians = 0
        print("Mean of Cuse value of all videos: " + str(cuse_videos_sum/total_videos))
        saveData("{}/{}/results_{}.txt".format(frames_dir, video, precision), "Mean of Cuse value of all videos: " + str(cuse_videos_sum/total_videos), '  ', '  ')
 
def Cov(frames_dir, gt_file, precision):
    total_videos = 0
    cos_videos_sum = 0
    matched_user_frames_sum = 0
    total_user_frames_sum = 0

    if os.path.exists(frames_dir) and os.path.exists(gt_file):
        video_list = os.listdir(frames_dir)
        video_list.sort()  # to guarantee order
        gt_list = os.listdir(gt_file)
        gt_list.sort()
        for video in video_list:
            gt_user_list = os.listdir("{}/{}/".format(gt_file, video))
            gt_user_list.sort()
            for gt_user in gt_user_list:
                dic_metrics = bipartiteGraph("{}/{}/".format(frames_dir, video),
                                                  "{}/{}/{}/".format(gt_file, video, gt_user),
                                                  "{}/{}/{}".format(frames_dir, video, "keyframe.txt"))
                dic_cut = {k: v for k, v in dic_metrics.items() if v >= precision}
                matched_user_frames_sum += len(dic_cut)
                total_user_frames_sum += len(os.listdir("{}/{}/{}/".format(gt_file, video, gt_user)))
            print("for video " + str(video) + " we got a Cov value of " + str(matched_user_frames_sum/total_user_frames_sum))
            cos_videos_sum += matched_user_frames_sum / total_user_frames_sum
            total_user_frames_sum = 0
            matched_user_frames_sum = 0
            total_videos += 1
        print("Mean of Cov value of all videos: " + str(cos_videos_sum/total_videos))
        saveData("{}/{}/results_{}.txt".format(frames_dir, video, precision), "Mean of Cov value of all videos: " + str(cos_videos_sum/total_videos), '  ', '  ')

def get_metrics(frames_dir, gt_file, precision):
  total_videos = 0
  # cusa variables
  cusa_videos_sum = 0
  median_sum_for_cusa = 0
  total_of_medians = 0
  # cov variables
  cov_videos_sum = 0
  matched_user_frames_sum = 0
  total_user_frames_sum = 0
  #F-measure
  matchTrueGT = 0
  lenGT = 0
  lenAS = 0
  matchPrecision = 0
  matchRecall = 0
  totalFmeasure = 0
  maxFmeasure = 0

  if os.path.exists(frames_dir) and os.path.exists(gt_file):
      video_list = os.listdir(frames_dir)
      video_list.sort()  # to guarantee order
      gt_list = os.listdir(gt_file)
      gt_list.sort()
      for video in video_list:
          gt_user_list = os.listdir("{}/{}/".format(gt_file, video))
          gt_user_list.sort()
          for gt_user in gt_user_list:
              dic_metrics, lenKf, lenGt = bipartiteGraph("{}/{}/".format(frames_dir, video),
                                                "{}/{}/{}/".format(gt_file, video, gt_user),
                                                "{}/{}/{}".format(frames_dir, video, "keyframe.txt"))
              dic_cut = {k: v for k, v in dic_metrics.items() if v >= precision}
              # cov calculation
              matched_user_frames_sum += len(dic_cut)
              total_user_frames_sum += len(os.listdir("{}/{}/{}/".format(gt_file, video, gt_user)))
              # cusa calculation
              median = len(dic_cut) / len(os.listdir("{}/{}/{}/".format(gt_file, video, gt_user)))
              median_sum_for_cusa += median
              total_of_medians += 1
              # F-measure
              matchTrueGT = len(dic_cut)
              lenGT = lenGt
              lenAS = lenKf
              matchPrecision = matchTrueGT / lenGT
              matchRecall = matchTrueGT / lenAS

              fmeasure = 2*matchPrecision*matchRecall/(matchRecall+matchPrecision)
              if(fmeasure > maxFmeasure):
                 maxFmeasure = fmeasure

          print("for video " + str(video) + " we got a f-measure value of " + str(
               fmeasure))

          saveData("{}/{}/cov_{}.txt".format(frames_dir, video, video), 
                     "for video " + str(video) + " we got a Cov value of " + str(
                      matched_user_frames_sum / total_user_frames_sum)
                     , "{}".format( precision), '  ')
          saveData("{}/{}/cusa_{}.txt".format(frames_dir, video, video), 
                     "for video " + str(video) + " we got a Cusa value of " + str(
                      median_sum_for_cusa / total_of_medians)
                     , "{}".format( precision), '  ')
          saveData("{}/{}/cuse_{}.txt".format(frames_dir, video, video), 
                     "for video " + str(video) + " we got a Cuse value of " + str(
                      1 - (median_sum_for_cusa / total_of_medians))
                     , "{}".format( precision), '  ')

          # cusa mean
          cusa_videos_sum += median_sum_for_cusa / total_of_medians
          total_videos += 1
          median_sum_for_cusa = 0
          total_of_medians = 0
          # cov mean
          cov_videos_sum += matched_user_frames_sum / total_user_frames_sum
          total_user_frames_sum = 0
          matched_user_frames_sum = 0
          # F-measure
          totalFmeasure += fmeasure
      print("Max of F-measure value: " + str(maxFmeasure))
      print("Mean of F-measure value of all videos: " + str(totalFmeasure / total_videos))
      print("Mean of Cov value of all videos: " + str(cov_videos_sum / total_videos))
      print("Mean of Cusa value of all videos: " + str(cusa_videos_sum / total_videos))
      print("Mean of Cuse value of all videos: " + str(1 - (cusa_videos_sum / total_videos)))

      saveData("{}/{}/results_{}.txt".format(frames_dir, video, precision), 
               "Mean of Cov value of all videos: " + str(cov_videos_sum / total_videos), '  ', '  ')

      saveData("{}/{}/results_{}.txt".format(frames_dir, video, precision), 
               "Mean of Cusa value of all videos: " + str(cusa_videos_sum / total_videos), '  ', '  ')
      
      saveData("{}/{}/results_{}.txt".format(frames_dir, video, precision), 
               "Mean of Cuse value of all videos: " + str(1 - (cusa_videos_sum / total_videos)), '  ', '  ')

def Cov2(frames_dir, gt_file, precision):
    total_videos = 0

    cos_videos_sum = 0
    matched_user_frames_sum = 0
    total_user_frames_sum = 0

    cusa_videos_sum = 0
    median_sum_for_cusa = 0
    
    cuse_videos_sum = 0
    median_sum_for_cuse= 0

    total_of_medians = 0

    if os.path.exists(frames_dir) and os.path.exists(gt_file):
        video_list = os.listdir(frames_dir)
        video_list.sort()  # to guarantee order
        gt_list = os.listdir(gt_file)
        gt_list.sort()
        for video in video_list:
            gt_user_list = os.listdir("{}/{}/".format(gt_file, video))
            gt_user_list.sort()
            for gt_user in gt_user_list:
                dic_metrics = bipartiteGraph("{}/{}/".format(frames_dir, video),
                                                  "{}/{}/{}/".format(gt_file, video, gt_user),
                                                  "{}/{}/{}".format(frames_dir, video, "keyframe.txt"))
                #print(dic_metrics)
                dic_cut = {k: v for k, v in dic_metrics.items() if v >= precision}
                matched_user_frames_sum += len(dic_cut)
                total_user_frames_sum += len(os.listdir("{}/{}/{}/".format(gt_file, video, gt_user)))

                ## use in cusa
                median = len(dic_cut)/len(os.listdir("{}/{}/{}/".format(gt_file, video, gt_user)))
                median_sum_for_cusa += median
                
                ## used in cuse
                median_sum_for_cuse += median
                total_of_medians += 1
            
            ## save cov by video
            print("for video " + str(video) + " we got a Cov value of " + str(matched_user_frames_sum/total_user_frames_sum))
            saveData("{}/{}/cov_{}.txt".format(frames_dir, video, video), 
                     "for video " + str(video) + " we got a Cov value of " + str(matched_user_frames_sum/total_user_frames_sum)
                     , "{}".format( precision), '  ')
            
            ## save cusa by video
            print("for video " + str(video) + " we got a Cusa value of " + str(median_sum_for_cusa/total_of_medians))
            saveData("{}/{}/cusa_{}.txt".format(frames_dir, video, video), 
                     "for video " + str(video) + " we got a Cusa value of " + str(median_sum_for_cusa/total_of_medians)
                     , "{}".format( precision), '  ')

            ## save cuse by video
            print("for video " + str(video) + " we got a Cuse value of " + str(median_sum_for_cuse/total_of_medians))
            saveData("{}/{}/cuse_{}.txt".format(frames_dir, video, video), 
                     "for video " + str(video) + " we got a Cuse value of " + str(median_sum_for_cuse/total_of_medians)
                     , "{}".format( precision), '  ')
            ## Cov Metric
            cos_videos_sum += matched_user_frames_sum / total_user_frames_sum
            total_user_frames_sum = 0
            matched_user_frames_sum = 0

            ## Cusa Metric
            cusa_videos_sum += median_sum_for_cusa/total_of_medians
            median_sum_for_cusa = 0
            
            ## Cuse Metric            
            cuse_videos_sum += median_sum_for_cuse/total_of_medians
            median_sum_for_cuse = 0
            total_of_medians = 0
            total_videos += 1

        print("Mean of Cov value of all videos: " + str(cos_videos_sum/total_videos))
        saveData("{}/results_{}.txt".format(frames_dir, precision), "Mean of Cov value of all videos: " + str(cos_videos_sum/total_videos), '  ', '  ')

        print("Mean of Cusa value of all videos: " + str(cusa_videos_sum/total_videos))
        saveData("{}/results_{}.txt".format(frames_dir, precision), "Mean of Cusa value of all videos: " + str(cusa_videos_sum/total_videos), '  ', '  ')
        
        print("Mean of Cuse value of all videos: " + str(cuse_videos_sum/total_videos))
        saveData("{}/results_{}.txt".format(frames_dir, precision), "Mean of Cuse value of all videos: " + str(cuse_videos_sum/total_videos), '  ', '  ')

def loadFrames(video_file, delta_t, output_graph_file, features_list):
  if(os.path.exists(video_file)):
    frame_list = os.listdir(video_file)
    frame_list.sort() # to garanted the time order

    feat_list_len = len(features_list)
    weight_list = []

    for vertex1 in range(feat_list_len):

      for vertex2 in range(calc_init(vertex1, delta_t, feat_list_len),
                                    calc_end(vertex1, delta_t, feat_list_len)):

        frame1 = os.path.join(video_file, frame_list[vertex1])
        frame2 = os.path.join(video_file, frame_list[vertex2])
        w = compute_similarity_img(model, frame1, frame2, features_list[vertex1], features_list[vertex2])

        weight_list.append(w)

        saveData(output_graph_file, "{}, {}, {:.2f}".format(vertex1, vertex2, w) , ' ', ' ')
        
    print(video_file)
    print(round(np.std(weight_list)) - 1)
        
if __name__ == "__main__":

    dataset_videos = sys.argv[1]
    dataset_frames = sys.argv[2]
    userSummary = sys.argv[3]

    rate = sys.argv[4]
    time = sys.argv[5]

    if(os.path.exists(dataset_videos)):
      video_list = os.listdir(dataset_videos)
      video_list.sort() # to guarantee order
      for i in video_list:
        print("------------------------")
        print("{}/{}".format(dataset_videos, i))
        frame_extractor("{}/{}".format(dataset_videos, i), rate)
    elif(os.file.exists(dataset_videos)):
      frame_extractor(dataset_videos, rate)
    else:
        print("do not exist directory or file with this path")

    if(os.path.exists(dataset_frames)):
      video_list = os.listdir(dataset_frames)
      video_list.sort() # to guarantee order
      for video in video_list:
        hierarchical_summarization("{}/{}/".format(dataset_frames, video), rate, time)
    
    get_metrics(dataset_frames, userSummary, int(sys.argv[1]))

    print("Done")