import numpy as np
import torch
from torchvision import transforms
import os
from PIL import Image
import math
import cv2
from mpmath import mp
from itertools import compress
from joblib import Parallel, delayed
import json
import pycocotools.mask as mask_util

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
    """
    Loads bounding boxes from JSON files in COCO format.
    
    Args:
    paths (list): List of file paths or image identifiers to load bounding boxes for.

    Returns:
    list: A list of dictionaries, each containing image metadata and its corresponding bounding boxes.
    """
    ground_truth = []

    # Iterate over each path
    for i, p in enumerate(paths):
        json_path = os.path.join("CraterLabels", f"{p}.json")  # Assuming JSON files are named after paths

        # Open and parse the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
            samples = {}
            samples['boxes'] = []

            # Extract bounding boxes from annotations
            for annotation in data['annotations']:
                bbox = annotation['bbox']
                x, y, w, h = bbox
                box = [int(x), int(y), int(x + w), int(y + h)]  # Convert bbox to [x_min, y_min, x_max, y_max]

                box = [max(min(coord, 1200), 0) for coord in box[:2]] + [max(min(coord, 1920), 0) for coord in box[2:]]
                
                samples['boxes'].append(box)

            ground_truth.append(samples)
    return ground_truth

def load_projected_ellipses(path):
    json_path = os.path.join("CraterLabels", f"{path}.json")
    samples = {'ellipse_sparse': []}

    # Check if the file exists
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return None

    try:
        with open(json_path) as f:
            data = json.load(f)
            # print("Loaded JSON data successfully.")
            # print(f"Number of annotations: {len(data.get('annotations', []))}")

            for annotation in data.get('annotations', []):
                # Check if the annotation has 'segmentation'
                # if 'segmentation' not in annotation or not isinstance(annotation['segmentation'], dict):
                #     print("No valid segmentation found in annotation. Skipping...")
                #     continue

                rle = annotation['segmentation']
                width = data['images'][0]['width']  # Access width of the first image
                height = data['images'][0]['height']  # Access height of the first image

                # Proceed only if height and width are valid
                if height <= 0 or width <= 0:
                    print(f"Invalid height or width in annotation: {height}, {width}. Skipping...")
                    continue

                # Decode RLE to binary mask
                size = [height, width]     # Example size (replace with your actual size)

                # Create the RLE dictionary
                cocoRLE = {
                    'counts': rle,
                    'size': size
                }
                binary_mask = mask_util.decode(cocoRLE)

                if np.sum(binary_mask) == 0:
                    print("Binary mask is empty for this annotation. Skipping...")
                    print(annotation)
                    continue

                # Find contours
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # If there are contours, fit an ellipse to the largest contour
                if contours and len(contours[0]) >= 5:
                    largest_contour = max(contours, key=cv2.contourArea)
                    ellipse = cv2.fitEllipse(largest_contour)
                    (x, y), (a, b), theta = ellipse
                    theta = np.radians(theta)

                    # Adjust theta as per your logic
                    if theta > np.pi / 2:
                        theta -= np.pi
                    elif theta < -np.pi / 2:
                        theta += np.pi

                    samples['ellipse_sparse'].append([x, y, a / 2, b / 2, theta])
                    # print(f"Ellipse fitted: {samples['ellipse_sparse'][-1]}")
                else:
                    # Fallback to bounding box
                    if 'bbox' in annotation:
                        x_min, y_min, width, height = annotation['bbox']
                        x = x_min + width / 2
                        y = y_min + height / 2
                        a = width / 2
                        b = height / 2
                        theta = 0  # Default to 0 rotation

                        samples['ellipse_sparse'].append([x, y, a, b, theta])
                        print(f"Ellipse fitted from bbox: {[x, y, a, b, theta]}")

    except json.JSONDecodeError:
        print(f"Error decoding JSON from: {json_path}")
        return None

    # Convert to tensor and check shape
    if samples['ellipse_sparse']:
        samples['ellipse_sparse'] = torch.tensor(samples['ellipse_sparse'], dtype=torch.float32)
        if len(samples['ellipse_sparse'].shape) < 2:
            print("No valid ellipse data found. Returning None.")
            return None
    else:
        print("No ellipses were fitted. Returning None.")
        return None

    # print("Final ellipse data:", samples['ellipse_sparse'])
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
            e[4].item() * 180 / math.pi, # Convert angle from radians to degrees
            0, # Start Angle for drawing
            360, # End Angle for drawing
            ( 1 ),
            -1, # Mask should be filled
        )
        masks.append( img )
    masks = np.array( masks )
    return torch.tensor( masks, dtype = torch.uint8 )

def load_crater_ids(paths):
    """
    Load crater IDs from JSON files in COCO format.

    Args:
    paths (list): List of file paths or image identifiers to load crater IDs for.

    Returns:
    np.array: A numpy array of dictionaries, each containing the crater IDs.
    """
    ground_truth = []

    # Iterate over each path
    for i, p in enumerate(paths):
        json_path = os.path.join("CraterLabels", f"{p}.json")  # Assuming JSON files are named after paths

        # Open and parse the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
            samples = {}
            samples['crater_id'] = []

            # Extract crater IDs from annotations (assuming each annotation corresponds to a crater)
            for annotation in data['annotations']:
                crater_id = annotation['id']  # Assuming 'id' refers to the crater ID in the annotation
                samples['crater_id'].append(crater_id)

            ground_truth.append(samples)

    return np.array(ground_truth)

class CraterDataset( torch.utils.data.Dataset ):
    def __init__( self, catalogue, img_size ):
        self.root = 'Craters/'
        self.imgs = []
        self.img_size = img_size
        self.catalogue = catalogue
        imgs = list(sorted(os.listdir(self.root)))
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
        
        #  8/9 split (Train - Test)
        splitIndex = totalSamples * 8 // 9  # Index to split at 75%
        
        # Indices for training and testing
        trainIndices = list(range(0, splitIndex))
        testIndices = list(range(splitIndex, totalSamples))
        
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
        mask = np.array( np.ones( len( bboxes ) ), dtype = np.bool )
        if np.sum( mask ) > 0: mask = np.logical_and( self.sizeFilter( bboxes, minArea = 0.1 ), mask )
        # if np.sum( mask ) > 0: mask = np.logical_and( self.depthFilter( ids, minDepth = 0.2 ), mask )
        
        target = {}
        target['boxes'] = bboxes[mask]
        # print(self.imgs[idx])
        # print (bboxes.shape)
        # print(len(ellipses['ellipse_sparse']))
        target['ellipse_sparse'] = ellipses['ellipse_sparse'][mask]
        target['labels'] = labels[mask]
        target['masks'] = compute_mask( self.img_size, target['ellipse_sparse'] )
        target['crater_id'] = ids[mask]
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
        min_mask = np.array( areas >= minArea, dtype = np.bool )
        max_mask = np.array( areas <= maxArea, dtype = np.bool )
        return np.logical_and( min_mask, max_mask )
    
    def depthFilter( self, crater_ids, minDepth = 0.2 ):
        depths = self.catalogue.loc[self.catalogue['CRATER_ID'].isin( crater_ids )]['medianDifference']
        return np.array( depths >= minDepth, dtype = np.bool )
    
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
    # a_m = ellipse_to_conic_matrix( *wrap_ellipse( [ a[2], a[3], a[0], a[1], a[4], ] ) ).float()
    # b_m = ellipse_to_conic_matrix( *wrap_ellipse( [ b[2], b[3], b[0], b[1], b[4], ] ) ).float()
    
    # error['gaussian_angle'] = gaussian_angle_distance( a_m, b_m ).item()
    # error['kl_divergence'] = norm_mv_kullback_leibler_divergence( a_m.unsqueeze( 0 ), b_m.unsqueeze( 0 ) ).item()
    
    # Intersection over union!
    img_shape = ( 1200, 1920, 3 )
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