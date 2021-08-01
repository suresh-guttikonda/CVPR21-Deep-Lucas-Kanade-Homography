
import sys
sys.path.append('../')

from data_read import *
from net import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
import random
import os





parser = argparse.ArgumentParser()


parser.add_argument('--dataset_name', action="store", dest= "dataset_name",default="GoogleMap",help='MSCOCO,GoogleMap,GoogleEarth,DayNight')


parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.00001,help='learning_rate')

parser.add_argument('--batch_size', action="store", dest="batch_size", type=int, default=4,help='batch_size')

parser.add_argument('--feature_map_type', action="store", dest="feature_map_type", default='special',help='regular or special')


parser.add_argument('--save_eval_f', action="store", dest="save_eval_f", type=int, default=400000,help='save and eval after how many iterations')

parser.add_argument('--epoch_load_one', action="store", dest="epoch_load_one", type=int, default=10,help='load the epoch number from level one')

parser.add_argument('--epoch_load_two', action="store", dest="epoch_load_two", type=int, default=10,help='load the epoch number from level two')


parser.add_argument('--sample_noise', action="store", dest="sample_noise", type=int, default=4,help='samples noise number')

parser.add_argument('--lambda_loss', action="store", dest="lambda_loss", type=float, default=0.2,help='0.2 for Google')



parser.add_argument('--epoch_start', action="store", dest="epoch_start", type=int, default=1,help='train from which epoch')


parser.add_argument('--epoch_num', action="store", dest="epoch_num", type=int, default=10,help='how many epochs to train')


input_parameters = parser.parse_args()
input_parameters.seed = 42


random.seed(input_parameters.seed)
np.random.seed(input_parameters.seed)
tf.random.set_seed(input_parameters.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)



if input_parameters.feature_map_type=='regular':
    load_path_one='./checkpoints/'+input_parameters.dataset_name+'/level_one_regular/'

    load_path_two='./checkpoints/'+input_parameters.dataset_name+'/level_two_regular/'

    save_path='./checkpoints/'+input_parameters.dataset_name+'/level_three_regular/'

elif input_parameters.feature_map_type=='special':


    load_path_one='./checkpoints/'+input_parameters.dataset_name+'/level_one/'

    load_path_two='./checkpoints/'+input_parameters.dataset_name+'/level_two/'

    save_path='./checkpoints/'+input_parameters.dataset_name+'/level_three/'




if not(os.path.exists('./checkpoints')):
    os.mkdir('./checkpoints')
if not(os.path.exists('./checkpoints/'+input_parameters.dataset_name)):
    os.mkdir('./checkpoints/'+input_parameters.dataset_name)
if not(os.path.exists(save_path)):
    os.mkdir(save_path)


lr=input_parameters.learning_rate



if input_parameters.feature_map_type=='regular':
    level_one_input=ResNet_first_input(if_regular=True)
    level_one_template=ResNet_first_template(if_regular=True)
    level_two_input=ResNet_second_input(if_regular=True)
    level_two_template=ResNet_second_template(if_regular=True)
    level_three_input=ResNet_third_input(if_regular=True)
    level_three_template=ResNet_third_template(if_regular=True)

elif input_parameters.feature_map_type=='special':
    level_one_input=ResNet_first_input()
    level_one_template=ResNet_first_template()
    level_two_input=ResNet_second_input()
    level_two_template=ResNet_second_template()
    level_three_input=ResNet_third_input()
    level_three_template=ResNet_third_template()


level_one_input.load_weights(load_path_one + 'epoch_'+str(input_parameters.epoch_load_one)+"input_full")

level_one_template.load_weights(load_path_one + 'epoch_'+str(input_parameters.epoch_load_one)+"template_full")



level_two_input.load_weights(load_path_two + 'epoch_'+str(input_parameters.epoch_load_two)+"input_full")

level_two_template.load_weights(load_path_two + 'epoch_'+str(input_parameters.epoch_load_two)+"template_full")




if input_parameters.epoch_start>1:
    #load weights
    level_three_input.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_start-1)+"input_full")

    level_three_template.load_weights(save_path + 'epoch_'+str(input_parameters.epoch_start-1)+"template_full")




def initial_motion_COCO():
    # prepare source and target four points
    matrix_list=[]
    for i in range(input_parameters.batch_size):

        src_points=[[0,0],[127,0],[127,127],[0,127]]

        tgt_points=[[32,32],[160,32],[160,160],[32,160]]


        src_points=np.reshape(src_points,[4,1,2])
        tgt_points=np.reshape(tgt_points,[4,1,2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points,0)
        matrix_list.append(h_matrix)
    return np.asarray(matrix_list).astype(np.float32)


def construct_matrix(initial_matrix,scale_factor,batch_size):
    #scale_factor size_now/(size to get matrix)
    initial_matrix=tf.cast(initial_matrix,dtype=tf.float32)

    scale_matrix=np.eye(3)*scale_factor
    scale_matrix[2,2]=1.0
    scale_matrix=tf.cast(scale_matrix,dtype=tf.float32)
    scale_matrix_inverse=tf.linalg.inv(scale_matrix)

    scale_matrix=tf.expand_dims(scale_matrix,axis=0)
    scale_matrix=tf.tile(scale_matrix,[batch_size,1,1])

    scale_matrix_inverse=tf.expand_dims(scale_matrix_inverse,axis=0)
    scale_matrix_inverse=tf.tile(scale_matrix_inverse,[batch_size,1,1])

    final_matrix=tf.matmul(tf.matmul(scale_matrix,initial_matrix),scale_matrix_inverse)
    return final_matrix



def average_cornner_error(batch_size,predicted_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127):

    four_conner=[[top_left_u,top_left_v,1],[bottom_right_u,top_left_v,1],[bottom_right_u,bottom_right_v,1],[top_left_u,bottom_right_v,1]]
    four_conner=np.asarray(four_conner)
    four_conner=np.transpose(four_conner)
    four_conner=np.expand_dims(four_conner,axis=0)
    four_conner=np.tile(four_conner,[batch_size,1,1]).astype(np.float32)

    new_four_points=tf.matmul(predicted_matrix,four_conner)

    new_four_points_scale=new_four_points[:,2:,:]
    new_four_points= new_four_points/new_four_points_scale


    u_predict=new_four_points[:,0,:]
    v_predict=new_four_points[:,1,:]

    average_conner=tf.math.pow(u_predict-u_list,2)+tf.math.pow(v_predict-v_list,2)
    #print (np.shape(average_conner))
    average_conner=tf.reduce_sum(average_conner)/batch_size


    return average_conner



'''
def compute_ssim(img_1,img_2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    paddings = tf.constant([[0,0],[1, 1,], [1, 1],[0,0]])

    img_1=tf.pad(img_1, paddings, "REFLECT")
    img_2=tf.pad(img_2, paddings, "REFLECT")

    mu_1=tf.nn.avg_pool2d(img_1,ksize=3,strides=1,padding='VALID')
    mu_2=tf.nn.avg_pool2d(img_2,ksize=3,strides=1,padding="VALID")

    sigma_1=tf.nn.avg_pool2d(img_1**2,ksize=3,strides=1,padding='VALID')-mu_1**2
    sigma_2=tf.nn.avg_pool2d(img_2**2,ksize=3,strides=1,padding='VALID')-mu_2**2
    sigma_1_2=tf.nn.avg_pool2d(img_1*img_2,ksize=3,strides=1,padding='VALID')-mu_1*mu_2

    SSIM_n=(2 * mu_1 * mu_2 + C1) * (2 * sigma_1_2 + C2)
    SSIM_d = (mu_1 ** 2 + mu_2 ** 2 + C1) * (sigma_1 + sigma_2 + C2)

    #return (1 - SSIM_n / SSIM_d) / 2

    return tf.clip_by_value((1 - SSIM_n / SSIM_d) / 2, 0, 1)
'''

def compute_ssim(img_1,img_2):

    return tf.math.pow((img_1-img_2),2)


def gt_motion_rs(u_list,v_list,batch_size=1):
    # prepare source and target four points
    matrix_list=[]
    for i in range(batch_size):

        src_points=[[0,0],[127,0],[127,127],[0,127]]

        #tgt_points=[[32*2+1,32*2+1],[160*2+1,32*2+1],[160*2+1,160*2+1],[32*2+1,160*2+1]]

        tgt_points=np.concatenate([u_list[i:(i+1),:],v_list[i:(i+1),:]],axis=0)
        tgt_points=np.transpose(tgt_points)
        tgt_points=np.expand_dims(tgt_points,axis=1)

        src_points=np.reshape(src_points,[4,1,2])
        tgt_points=np.reshape(tgt_points,[4,1,2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points,0)

        matrix_list.append(h_matrix)
    return np.asarray(matrix_list).astype(np.float32)

def gt_motion_rs_random_noisy(u_list,v_list,batch_size,lambda_noisy):
    # prepare source and target four points
    matrix_list=[]
    for i in range(batch_size):

        src_points=[[0,0],[127,0],[127,127],[0,127]]

        #tgt_points=[[32*2+1,32*2+1],[160*2+1,32*2+1],[160*2+1,160*2+1],[32*2+1,160*2+1]]

        tgt_points=np.concatenate([u_list[i:(i+1),:],v_list[i:(i+1),:]],axis=0)
        tgt_points=np.transpose(tgt_points)
        tgt_points=np.expand_dims(tgt_points,axis=1)

        src_points=np.reshape(src_points,[4,1,2])
        tgt_points=np.reshape(tgt_points,[4,1,2])

        # find homography
        h_matrix, status = cv2.findHomography(src_points, tgt_points,0)
        element_h_matrix=np.reshape(h_matrix,(9,1))
        noisy_matrix=np.zeros((9,1))
        for jj in range(8):
            #if jj!=0 and jj!=4:
            noisy_matrix[jj,0]=element_h_matrix[jj,0]*lambda_noisy[jj]
        noisy_matrix=np.reshape(noisy_matrix,(3,3))
        matrix_list.append(noisy_matrix)
    return np.asarray(matrix_list).astype(np.float32)
'''
def calculate_feature_map(input_tensor):
    bs,height,width,channel=tf.shape(input_tensor)
    path_extracted=tf.image.extract_patches(input_tensor, sizes=(1,3,3,1), strides=(1,1,1,1), rates=(1,1,1,1), padding='SAME')
    path_extracted=tf.reshape(path_extracted,(bs,height,width,channel,9))
    path_extracted_mean=tf.math.reduce_mean(path_extracted,axis=3,keepdims=True)

    #path_extracted_mean=tf.tile(path_extracted_mean,[1,1,1,channel,1])
    path_extracted=path_extracted-path_extracted_mean
    path_extracted_transpose=tf.transpose(path_extracted,(0,1,2,4,3))
    variance_matrix=tf.matmul(path_extracted_transpose,path_extracted)
    eigenvalue=tf.linalg.eigh(variance_matrix)[0]
    return  tf.math.reduce_max(eigenvalue,axis=-1,keepdims=True)/tf.math.reduce_sum(eigenvalue,axis=-1,keepdims=True)
'''

def calculate_feature_map(input_tensor):
    bs,height,width,channel=tf.shape(input_tensor)
    path_extracted=tf.image.extract_patches(input_tensor, sizes=(1,3,3,1), strides=(1,1,1,1), rates=(1,1,1,1), padding='SAME')
    path_extracted=tf.reshape(path_extracted,(bs,height,width,channel,9))
    path_extracted_mean=tf.math.reduce_mean(path_extracted,axis=3,keepdims=True)

    #path_extracted_mean=tf.tile(path_extracted_mean,[1,1,1,channel,1])
    path_extracted=path_extracted-path_extracted_mean
    path_extracted_transpose=tf.transpose(path_extracted,(0,1,2,4,3))
    variance_matrix=tf.matmul(path_extracted_transpose,path_extracted)

    tracevalue=tf.linalg.trace(variance_matrix)
    row_sum=tf.reduce_sum(variance_matrix,axis=-1)
    max_row_sum=tf.math.reduce_max(row_sum,axis=-1)
    min_row_sum=tf.math.reduce_min(row_sum,axis=-1)
    mimic_ratio=(max_row_sum+min_row_sum)/2.0/tracevalue

    return  tf.expand_dims(mimic_ratio,axis=-1)



initial_matrix=initial_motion_COCO()

LK_layer_three=Lucas_Kanade_layer(batch_size=input_parameters.batch_size,height_template=32,width_template=32,num_channels=1)

# Logging
summaries_flush_secs=10
summary_writer = tf.compat.v2.summary.create_file_writer(
    './log/level_one', flush_millis=summaries_flush_secs * 1000)


#initial_matrix_scaled=construct_matrix(initial_matrix,scale_factor=0.125,batch_size=input_parameters.batch_size)


for current_epoch in range(input_parameters.epoch_num):


    if input_parameters.dataset_name=='MSCOCO':
        data_loader_caller=data_loader_MSCOCO('train')

    if input_parameters.dataset_name=='GoogleMap':
        data_loader_caller=data_loader_GoogleMap('train')


    if input_parameters.dataset_name=='GoogleEarth':
        # data_loader_caller=data_loader_GoogleEarth('train')
        data_loader_caller=data_loader_Akagi('train')

    if input_parameters.dataset_name=='DayNight':
        data_loader_caller=data_loader_DayNight('train')



    optimizer = tf.keras.optimizers.Adam(lr=lr,beta_1=0.9)

    print("Starting epoch " + str(current_epoch+input_parameters.epoch_start))
    print("Learning rate is " + str(lr))

    error_ave_1000=0.0
    convex_loss_total=0.0
    ssim_loss_total=0.0

    for iters in range(10000000):
        input_img,u_list,v_list,template_img=data_loader_caller.data_read_batch(batch_size=input_parameters.batch_size)

        if len(np.shape(input_img))<2:
            if current_epoch%input_parameters.save_eval_f==0:
                level_three_input.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"input_full")
                level_three_template.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"template_full")
            break

        input_feature_one=level_one_input.call(input_img,training=False)
        template_feature_one=level_one_template.call(template_img,training=False)

        input_feature_two=level_two_input.call(input_feature_one,training=False)
        template_feature_two=level_two_template.call(template_feature_one,training=False)

        gt_matrix=gt_motion_rs(u_list,v_list,batch_size=input_parameters.batch_size)
        gt_matrix_three=construct_matrix(gt_matrix,scale_factor=0.25,batch_size=input_parameters.batch_size)

        with tf.GradientTape() as tape:


            input_feature=level_three_input.call(input_feature_two)
            template_feature=level_three_template.call(template_feature_two)



            if input_parameters.feature_map_type=='regular':
                input_feature_map_three=input_feature
                template_feature_map_three=template_feature

            elif input_parameters.feature_map_type=='special':
                input_feature_map_three=calculate_feature_map(input_feature)
                template_feature_map_three=calculate_feature_map(template_feature)


            input_warped_to_template_three=LK_layer_three.projective_inverse_warp(input_feature_map_three, gt_matrix_three)

            ssim_middle_three=tf.reduce_mean(compute_ssim(template_feature_map_three,input_warped_to_template_three))

            ssim_middle=ssim_middle_three



            #print ('!!!!!!!!!')

            for nn in range(input_parameters.sample_noise):
                lambda_three=(np.random.rand(8)-0.5)/6.0

                for mm in range(len(lambda_three)):
                    if lambda_three[mm]>0 and lambda_three[mm]<0.02:
                        lambda_three[mm]=0.02
                    if lambda_three[mm]<0 and lambda_three[mm]>-0.02:
                        lambda_three[mm]=-0.02



                gt_matrix_noisy_three=gt_motion_rs_random_noisy(u_list,v_list,batch_size=input_parameters.batch_size,lambda_noisy=lambda_three)
                gt_matrix_noisy_three=construct_matrix(gt_matrix_noisy_three,scale_factor=0.25,batch_size=input_parameters.batch_size)






                input_warped_to_template_shift_three_left=LK_layer_three.projective_inverse_warp(input_feature_map_three, gt_matrix_three+gt_matrix_noisy_three)
                ssim_shift_three_left=tf.reduce_mean(compute_ssim(template_feature_map_three,input_warped_to_template_shift_three_left))
                ssim_convex_three_left= -tf.math.minimum((ssim_shift_three_left-ssim_middle_three)-np.sum(lambda_three**2),0)

                '''
                input_warped_to_template_shift_three_left_left=LK_layer_three.projective_inverse_warp(input_feature_map_three, gt_matrix_three+1.5*gt_matrix_noisy_three)
                ssim_shift_three_left_left=tf.reduce_mean(compute_ssim(template_feature_map_three,input_warped_to_template_shift_three_left_left))

                input_warped_to_template_shift_three_left_right=LK_layer_three.projective_inverse_warp(input_feature_map_three, gt_matrix_three+0.5*gt_matrix_noisy_three)
                ssim_shift_three_left_right=tf.reduce_mean(compute_ssim(template_feature_map_three,input_warped_to_template_shift_three_left_right))

                ssim_convex_three_left_further= -tf.math.minimum((ssim_shift_three_left_left+ssim_shift_three_left_right-2*ssim_shift_three_left)-(np.sum((0.5*lambda_three)**2)+np.sum((1.5*lambda_three)**2)-2*np.sum(lambda_three**2)),0)
                '''
                if input_parameters.dataset_name=='MSCOCO':
                    ssim_convex_three_left_further=0
                else:

                    input_warped_to_template_shift_three_left_left=LK_layer_three.projective_inverse_warp(input_feature_map_three, gt_matrix_three+2.0*gt_matrix_noisy_three)
                    ssim_shift_three_left_left=tf.reduce_mean(compute_ssim(template_feature_map_three,input_warped_to_template_shift_three_left_left))


                    ssim_convex_three_left_further= -tf.math.minimum((ssim_shift_three_left_left-ssim_shift_three_left)-(np.sum((2*lambda_three)**2)-np.sum(lambda_three**2)),0)



                input_warped_to_template_shift_three_right=LK_layer_three.projective_inverse_warp(input_feature_map_three, gt_matrix_three-gt_matrix_noisy_three)
                ssim_shift_three_right=tf.reduce_mean(compute_ssim(template_feature_map_three,input_warped_to_template_shift_three_right))
                ssim_convex_three_right= -tf.math.minimum((ssim_shift_three_right-ssim_middle_three)-np.sum(lambda_three**2),0)


                '''
                input_warped_to_template_shift_three_right_left=LK_layer_three.projective_inverse_warp(input_feature_map_three, gt_matrix_three-1.5*gt_matrix_noisy_three)
                ssim_shift_three_right_left=tf.reduce_mean(compute_ssim(template_feature_map_three,input_warped_to_template_shift_three_right_left))

                input_warped_to_template_shift_three_right_right=LK_layer_three.projective_inverse_warp(input_feature_map_three, gt_matrix_three-0.5*gt_matrix_noisy_three)
                ssim_shift_three_right_right=tf.reduce_mean(compute_ssim(template_feature_map_three,input_warped_to_template_shift_three_right_right))

                ssim_convex_three_right_further= -tf.math.minimum((ssim_shift_three_right_left+ssim_shift_three_right_right-2*ssim_shift_three_right)-(np.sum((0.5*lambda_three)**2)+np.sum((1.5*lambda_three)**2)-2*np.sum(lambda_three**2)),0)

                '''
                if input_parameters.dataset_name=='MSCOCO':
                    ssim_convex_three_right_further=0
                else:
                    input_warped_to_template_shift_three_right_right=LK_layer_three.projective_inverse_warp(input_feature_map_three, gt_matrix_three-2.0*gt_matrix_noisy_three)
                    ssim_shift_three_right_right=tf.reduce_mean(compute_ssim(template_feature_map_three,input_warped_to_template_shift_three_right_right))


                    ssim_convex_three_right_further= -tf.math.minimum((ssim_shift_three_right_right-ssim_shift_three_right)-(np.sum((2*lambda_three)**2)-np.sum(lambda_three**2)),0)




                if nn==0:
                    convex_loss=ssim_convex_three_left+ssim_convex_three_right+ssim_convex_three_left_further+ssim_convex_three_right_further
                else:
                    convex_loss=convex_loss+ssim_convex_three_left+ssim_convex_three_right+ssim_convex_three_left_further+ssim_convex_three_right_further
            total_loss=ssim_middle+input_parameters.lambda_loss*convex_loss

            #print (ssim_middle)
            #print (convex_loss)

            convex_loss_total+=convex_loss
            ssim_loss_total+=ssim_middle
            error_ave_1000+=total_loss
            #print ('!!!!!!!!!!!!!!!!!')



        all_parameters=level_three_template.trainable_variables+level_three_input.trainable_variables

        grads = tape.gradient(total_loss, all_parameters)
        grads=[tf.clip_by_value(i,-0.1,0.1) for i in grads]
        optimizer.apply_gradients(zip(grads, all_parameters))




        with summary_writer.as_default():
            tf.summary.scalar('error_ave_1000', error_ave_1000, step=current_epoch)
            tf.summary.scalar('ssim_loss_total', ssim_loss_total, step=current_epoch)
            tf.summary.scalar('convex_loss_total', convex_loss_total, step=current_epoch)

            error_ave_1000=0.0
            convex_loss_total=0.0
            ssim_loss_total=0.0

        # if iters%input_parameters.save_eval_f==0 and iters>0:
        #
        #     level_three_input.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"input_"+str(iters))
        #     level_three_template.save_weights(save_path +'epoch_'+str(input_parameters.epoch_start+current_epoch)+"template_"+str(iters))



        input_img = None
        u_list = None
        v_list = None
        template_img = None

        input_feature_map = None
        template_feature_map = None
        input_warped_to_template=None
        input_warped_to_template_left_1=None
        input_warped_to_template_left_2=None
        input_warped_to_template_right_1=None
        input_warped_to_template_right_2=None
