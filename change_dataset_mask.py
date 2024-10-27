import numpy as np
import torch
from torchvision import transforms
import os
from PIL import Image
import math
import cv2
from mpmath import mp
from itertools import compress
from joblib import Parallel, delayed, load
import json
import pycocotools.mask as mask_util
from natsort import natsorted


labels = load('CE5-ellipse-labels')

def plot_ellipses( img, bboxes, ellipse_matrices, color = ( 1, 0, 0 ) ):
    for bbox, ellipse_matrix in zip( bboxes, ellipse_matrices ):
        # Plot bounding box:
        cv2.rectangle(
            img,
            ( int( bbox[0] ), int( bbox[1] ) ),
            ( int( bbox[2] ), int( bbox[3] ) ),
            color,
            2,
        )

        # Change this to use mask to predict ellipse!
        
        # ellipse_matrix = ellipse_matrix.numpy()
        # x, y = conic_center( ellipse_matrix )
        # a, b = ellipse_axes( ellipse_matrix )
        # theta = ellipse_angle( ellipse_matrix )
        
        # # Plot ellipse:
        # cv2.ellipse( 
        #     img,
        #     ( int( x.item() ), int( y.item() ) ), # Center point
        #     ( int( a.item() ), int( b.item() ) ), # Major and minor axes
        #     theta.item() * 180 / math.pi, # Convert angle from radians to degrees
        #     0, # Start Angle for drawing
        #     360, # End Angle for drawing
        #     color,
        #     2,
        # )
    return img

def load_bounding_boxes(paths):
    ground_truth = []
    for path in paths:
      for label in labels:
          # Only process the label if its 'id' is in the specified paths
          if label['id'] == path:
              samples = {}
              samples['boxes'] = []
              
              # Loop over all ellipses in the 'ellipse_sparse' data
              for ellipse in label['ellipse_sparse']:
                  x_center, y_center, a, b, angle = ellipse
                  angle = angle * 180 / math.pi
                  cos_angle = np.cos(angle)
                  sin_angle = np.sin(angle)
                  
                  # Calculate the bounding box
                  width = np.sqrt((a * cos_angle) ** 2 + (b * sin_angle) ** 2)
                  height = np.sqrt((a * sin_angle) ** 2 + (b * cos_angle) ** 2)
                  x_min = x_center - width
                  x_max = x_center + width
                  y_min = y_center - height
                  y_max = y_center + height
                  
                  # Add the bounding box to samples
                  samples['boxes'].append([x_min * 23.52, y_min * 17.28, x_max *23.52, y_max *17.28])
              
              # Append the samples for this label to the ground truth list
              ground_truth.append(samples)
    
    return ground_truth

def load_projected_ellipses(path):
    samples = {'ellipse_sparse': []}

    for label in labels:
        # Only process the label if its 'id' is in the specified paths
        if label['id'] == path:      
            # Loop over all ellipses in the 'ellipse_sparse' data
            for ellipse in label['ellipse_sparse']:
                x_center, y_center, semi_major, semi_minor, angle = ellipse

                # Scale the ellipse coordinates and sizes according to the image dimensions
                x_center_scaled = x_center * 23.52
                y_center_scaled = y_center * 17.28
                semi_major_scaled = semi_major * 23.52
                semi_minor_scaled = semi_minor * 17.28

                # Append the scaled ellipse parameters to the samples
                samples['ellipse_sparse'].append([x_center_scaled, y_center_scaled, semi_major_scaled, semi_minor_scaled, angle* math.pi / 180])

    if samples['ellipse_sparse']:
        samples['ellipse_sparse'] = torch.tensor(samples['ellipse_sparse'], dtype=torch.float32)
        if len(samples['ellipse_sparse'].shape) < 2:
            print("No valid ellipse data found. Returning None.")
            return None
    else:
        print("No ellipses were fitted. Returning None.")
        return None

    return samples

def compute_mask( image_size, ellipses ):
    '''
    Compute mask from ellipse and bbox
    Masks must be of the shape ( N, 1, H, W )
    N : Number of instances
    H : Image height
    W : Image width
    Returns masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    image_size is tuple ( H, W )
    ellipses is list where each element is [ x, y, a, b, theta ] 
    '''

    masks = []
    for e in ellipses:
        img = np.zeros( image_size )
        
        cv2.ellipse(
            img,
            ( int( e[0].item() ), int( e[1].item() ) ), # Center point
            ( int( e[2].item() ), int( e[3].item() ) ), # Major and minor axes
            e[4].item()* 180 / math.pi, # Convert angle from radians to degrees
            0, # Start Angle for drawing
            360, # End Angle for drawing
            ( 1 ),
            -1, # Mask should be filled
        )
        masks.append( img )
    masks = np.array( masks )
    return torch.tensor( masks, dtype = torch.uint8 )

def load_crater_ids(paths):
    ground_truth = []
    samples = {}
    samples['crater_id'] = []
    for path in paths :
        samples['crater_id'].append(path.replace('.png', ''))

    ground_truth.append(samples)
    return np.array(ground_truth)

class CraterDataset( torch.utils.data.Dataset ):
    def __init__( self, catalogue, img_size ):
        self.root = 'CH5-png/'
        self.imgs = []
        self.img_size = img_size
        self.catalogue = catalogue
        imgs = list(natsorted(os.listdir(self.root)))
        imgs = [img for img in imgs if 'ipynb_checkpoints' not in img and '.DS_Store' not in img and img.endswith('.png')]

        # Add the images to self.imgs with the correct file paths
        self.imgs.extend([img for img in imgs])
        # Iterate through dataset and check for bad samples
        print( 'Filter bad samples:' )
        mask = Parallel( n_jobs = 16 )( delayed( checkSample )( *[ self, i ] ) for i in range( self.__len__() ) )
        # mask = []
        # for i in range( self.__len__() ):
        #     mask.append( checkSample( self, i ) )
        
        self.imgs = list( compress( self.imgs, mask ) )

        print( 'Total Images after Filtering', self.__len__() )
    
    def __len__( self ):
        return len( self.imgs )
    
    # Stratified split across each viewing angle
    # Return indices for img paths to test on
    def testSplit(self):
        # Get the total number of images
        totalSamples = len(self.imgs)

        # Initialize lists for training and testing indices
        testIndices = []
        trainIndices = []

        # Iterate through all the indices
        for i in range(totalSamples):
            testIndices.append(i)

        return trainIndices, testIndices

    def getTarget( self, idx ):
        # Get bounding boxes
        bboxes = load_bounding_boxes( [self.imgs[idx]] )[0]
        ellipses = load_projected_ellipses( self.imgs[idx] )
        ids = load_crater_ids( [self.imgs[idx]] )[0]

        bboxes = torch.as_tensor( np.array( bboxes['boxes'] ), dtype = torch.float32 )
        labels = torch.ones( len( bboxes ), dtype = torch.int64 )
        # depths = torch.as_tensor( np.array( self.getDepthsFromIDs( ids ) ), dtype = torch.float32 )
        ids = np.array( ids['crater_id'] )
        
        if len( bboxes.shape ) < 2:
            return None

        # Filter as necessary
        mask = np.array( np.ones( len( bboxes ) ), dtype = bool )
        if np.sum( mask ) > 0: mask = np.logical_and( self.sizeFilter( bboxes, minArea = 0.1 ), mask )
        # if np.sum( mask ) > 0: mask = np.logical_and( self.depthFilter( ids, minDepth = 0.2 ), mask )
        
        target = {}
        target['boxes'] = bboxes[mask]
        # print(self.imgs[idx])
        # print (bboxes.shape)
        # print(len(ellipses['ellipse_sparse']))
        # print(ids)
        target['ellipse_sparse'] = ellipses['ellipse_sparse'][mask]
        target['labels'] = labels[mask]
        target['masks'] = compute_mask( self.img_size, target['ellipse_sparse'] )
        target['image_id'] = self.root + self.imgs[idx]
        # target['depths'] = depths
        # target['view_angle'] = torch.tensor( [ self.img_angles[idx] ] )
        target['area'] = ( bboxes[:,3] - bboxes[:,1] ) * ( bboxes[:,2] - bboxes[:,0] )
        
        return target
        
    def __getitem__( self, idx ):
        # Load Image
        img = Image.open( self.root + self.imgs[idx] ).convert('L')
        transform = transforms.ToTensor()
        img = transform( img )
        
        target = self.getTarget( idx )
        return img, target

    def sizeFilter( self, bboxes, minArea = 25, maxArea = 2560 * 2560 ):
        areas = ( bboxes[:,3] - bboxes[:,1] ) * ( bboxes[:,2] - bboxes[:,0] )
        min_mask = np.array( areas >= minArea, dtype = bool )
        max_mask = np.array( areas <= maxArea, dtype = bool )
        return np.logical_and( min_mask, max_mask )

    def depthFilter( self, crater_ids, minDepth = 0.2 ):
        depths = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['medianDifference']
        return np.array( depths >= minDepth, dtype = bool )

def checkSample( dataset, i ):
    _, targets = dataset.__getitem__( i )
    if targets is None:
        return False
    return True
def evaluate_ellipse( a, b ):
    '''
    Evaluate error between two ellipses based on:
    - KL Divergence
    - Gaussian Angle
    - Intersection over Union
    - Absolute error in ellipse parameters
    
    Arguments:
    'a' and 'b' are lists such that:
    [ x centre, y centre, semimajor axis, semiminor axis, angle (radians) ]
    '''
    
    error = {}
    error['x_error'] = abs( a[0] - b[0] )
    error['y_error'] = abs( a[1] - b[1] )
    error['a_error'] = abs( a[2] - b[2] )
    error['b_error'] = abs( a[3] - b[3] )
    error['theta_error'] = abs( a[4] - b[4] )
    error['absolute_error'] = np.sum( np.abs( np.array( a ) - np.array( b ) ) )
    
    # Convert sparse ellipse params into conic matrices
    # Convert ellipse parameters to conic matrices inline
    cos_a, sin_a = np.cos(a[4]), np.sin(a[4])
    A_a = (cos_a**2) / a[2]**2 + (sin_a**2) / a[3]**2
    B_a = 2 * cos_a * sin_a * (1 / a[2]**2 - 1 / a[3]**2)
    C_a = (sin_a**2) / a[2]**2 + (cos_a**2) / a[3]**2
    D_a = -2 * A_a * a[0] - B_a * a[1]
    E_a = -2 * C_a * a[1] - B_a * a[0]
    F_a = A_a * a[0]**2 + B_a * a[0] * a[1] + C_a * a[1]**2 - 1
    a_m = torch.tensor([[A_a, B_a / 2, D_a / 2],
                        [B_a / 2, C_a, E_a / 2],
                        [D_a / 2, E_a / 2, F_a]]).float()

    cos_b, sin_b = np.cos(b[4]), np.sin(b[4])
    A_b = (cos_b**2) / b[2]**2 + (sin_b**2) / b[3]**2
    B_b = 2 * cos_b * sin_b * (1 / b[2]**2 - 1 / b[3]**2)
    C_b = (sin_b**2) / b[2]**2 + (cos_b**2) / b[3]**2
    D_b = -2 * A_b * b[0] - B_b * b[1]
    E_b = -2 * C_b * b[1] - B_b * b[0]
    F_b = A_b * b[0]**2 + B_b * b[0] * b[1] + C_b * b[1]**2 - 1
    b_m = torch.tensor([[A_b, B_b / 2, D_b / 2],
                        [B_b / 2, C_b, E_b / 2],
                        [D_b / 2, E_b / 2, F_b]]).float()

    # Calculate Gaussian angle distance inline
    error['gaussian_angle'] = torch.norm(a_m - b_m).item()

    # Calculate KL divergence inline
    det_a = torch.det(a_m[:2, :2])
    det_b = torch.det(b_m[:2, :2])
    trace_term = torch.trace(torch.mm(torch.inverse(b_m[:2, :2]), a_m[:2, :2]))
    error['kl_divergence'] = 0.5 * (torch.log(det_b / det_a) - 2 + trace_term).item()

    # Intersection over union!
    img_shape = ( 1728, 2352, 3 )
    img1 = np.zeros( img_shape )
    img2 = np.zeros( img_shape )
    
    # Draw predicted ellipse in Red channel (filled)
    cv2.ellipse(
        img1,
        ( int( a[0] ), int( a[1] ) ), # Center point
        ( int( a[2] ), int( a[3] ) ), # Semiminor and Semimajor axes
        float( a[4] * 180 / math.pi ), # Angle (convert from radians to degrees)
        0, # Start Angle for drawing
        360, # End Angle for drawing
        ( 1, 0, 0 ),
        -1,
    )
    
    cv2.ellipse(
        img2,
        ( int( b[0] ), int( b[1] ) ), # Center point
        ( int( b[2] ), int( b[3] ) ), # Semiminor and Semimajor axes
        float( b[4] * 180 / math.pi ), # Angle (convert from radians to degrees)
        0, # Start Angle for drawing
        360, # End Angle for drawing
        ( 1, 0, 0 ),
        -1,
    )
    
    intersection = np.logical_and( img1[:,:,0], img2[:,:,0] )
    union = np.logical_or( img1[:,:,0], img2[:,:,0] )
    error['IoU'] = np.sum( intersection ) / np.sum( union )
    return error
