#!/usr/bin/env python3

import sys
sys.path.append('../')
from data_read import *
from net import *
import numpy as np
import random
import tensorflow as tf
import time
import argparse
import os
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_name',
        action="store",
        dest= "dataset_name",
        default="GoogleEarth",
        help='MSCOCO,GoogleMap,GoogleEarth,DayNight'
    )
    parser.add_argument(
        '--epoch_load_one',
        action="store",
        dest="epoch_load_one",
        type=int,
        default=10,
        help='epoch_load_one'
    )
    parser.add_argument(
        '--epoch_load_two',
        action="store",
        dest="epoch_load_two",
        type=int,
        default=10,
        help='epoch_load_two'
    )
    parser.add_argument(
        '--epoch_load_three',
        action="store",
        dest="epoch_load_three",
        type=int,
        default=10,
        help='epoch_load_three'
    )
    parser.add_argument(
        '--num_iters',
        action="store",
        dest="num_iters",
        type=int,
        default=50,
        help='num_iters'
    )
    parser.add_argument(
        '--feature_map_type',
        action="store",
        dest="feature_map_type",
        default='special',
        help='regular or special'
    )
    parser.add_argument(
        '--initial_type',
        action="store",
        dest="initial_type",
        default='multi_net',
        help='vanilla, simple_net, multi_net'
    )
    parser.add_argument(
        '--if_LK',
        action="store",
        dest="if_LK",
        default=True,
        help='True or False'
    )
    parser.add_argument(
        '--load_epoch_simplenet',
        action="store",
        dest="load_epoch_simplenet",
        default=100,
        help='load_epoch_simplenet'
    )
    parser.add_argument(
        '--load_epoch_multinet',
        action="store",
        dest="load_epoch_multinet",
        default=[100,100,80],
        help='load_epoch_multinet'
    )

    input_parameters = parser.parse_args()

    return input_parameters

class DLK_Homography(object):
    """
    """

    def __init__(self, input_parameters):
        """
        """

        self.input_parameters = input_parameters

        # networks to compute inital matrix
        if input_parameters.initial_type=='multi_net':
            save_path_one='./checkpoints/'+input_parameters.dataset_name+'/regression_stage_1/'
            save_path_two='./checkpoints/'+input_parameters.dataset_name+'/regression_stage_2/'
            save_path_three='./checkpoints/'+input_parameters.dataset_name+'/regression_stage_3/'
            self.regression_network_one=Net_first()
            self.regression_network_two=Net_second()
            self.regression_network_three=Net_third()

        self.regression_network_one.load_weights(save_path_one + 'epoch_'+str(input_parameters.load_epoch_multinet[0]))
        self.regression_network_two.load_weights(save_path_two + 'epoch_'+str(input_parameters.load_epoch_multinet[1]))
        self.regression_network_three.load_weights(save_path_three + 'epoch_'+str(input_parameters.load_epoch_multinet[2]))

        # network to extract feature maps
        if input_parameters.feature_map_type=='special':
            load_path_one='./checkpoints/'+input_parameters.dataset_name+'/level_one/'
            load_path_two='./checkpoints/'+input_parameters.dataset_name+'/level_two/'
            load_path_three='./checkpoints/'+input_parameters.dataset_name+'/level_three/'

            self.level_one_input=ResNet_first_input()
            self.level_one_template=ResNet_first_template()
            self.level_two_input=ResNet_second_input()
            self.level_two_template=ResNet_second_template()
            self.level_three_input=ResNet_third_input()
            self.level_three_template=ResNet_third_template()

        self.level_one_input.load_weights(load_path_one + 'epoch_'+str(input_parameters.epoch_load_one)+"input_full")
        self.level_one_template.load_weights(load_path_one + 'epoch_'+str(input_parameters.epoch_load_one)+"template_full")
        self.level_two_input.load_weights(load_path_two + 'epoch_'+str(input_parameters.epoch_load_two)+"input_full")
        self.level_two_template.load_weights(load_path_two  + 'epoch_'+str(input_parameters.epoch_load_two)+"template_full")
        self.level_three_input.load_weights(load_path_three + 'epoch_'+str(input_parameters.epoch_load_three)+"input_full")
        self.level_three_template.load_weights(load_path_three  + 'epoch_'+str(input_parameters.epoch_load_three)+"template_full")

        # networks for lukas kanade estimation
        # self.LK_layer_one=Lucas_Kanade_layer(batch_size=1,height_template=324,width_template=432,num_channels=1)
        # self.LK_layer_two=Lucas_Kanade_layer(batch_size=1,height_template=162,width_template=216,num_channels=1)
        # self.LK_layer_three=Lucas_Kanade_layer(batch_size=1,height_template=81,width_template=108,num_channels=1)
        # self.LK_layer_regression=Lucas_Kanade_layer(batch_size=1,height_template=324,width_template=432,num_channels=3)

        self.LK_layer_one=Lucas_Kanade_layer(batch_size=1,height_template=128,width_template=128,num_channels=1)
        self.LK_layer_two=Lucas_Kanade_layer(batch_size=1,height_template=64,width_template=64,num_channels=1)
        self.LK_layer_three=Lucas_Kanade_layer(batch_size=1,height_template=32,width_template=32,num_channels=1)
        self.LK_layer_regression=Lucas_Kanade_layer(batch_size=1,height_template=192,width_template=192,num_channels=3)


        # data set
        if input_parameters.dataset_name=='GoogleEarth':
            # self.data_loader_caller=data_loader_GoogleEarth('val')
            self.data_loader_caller=data_loader_Akagi('val')
            # self.data_loader_caller=data_loader_Manual()

    def average_cornner_error(self, batch_size,predicted_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127):

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

        average_conner=tf.reduce_mean(tf.sqrt(tf.math.pow(u_predict-u_list,2)+tf.math.pow(v_predict-v_list,2)))

        return average_conner, u_predict, v_predict

    def construct_matrix(self, initial_matrix,scale_factor,batch_size):
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

    def construct_matrix_regression(self, batch_size,network_output,network_output_2=[0]):
        extra=tf.ones((batch_size,1))
        predicted_matrix=tf.concat([network_output,extra],axis=-1)
        predicted_matrix=tf.reshape(predicted_matrix,[batch_size,3,3])
        if len(np.shape(network_output_2))>1:
            predicted_matrix_2=tf.concat([network_output_2,extra],axis=-1)
            predicted_matrix_2=tf.reshape(predicted_matrix_2,[batch_size,3,3])
        hh_matrix=[]
        for i in range(batch_size):
            if len(np.shape(network_output_2))>1:
                hh_matrix.append(np.linalg.inv(np.dot(predicted_matrix_2[i,:,:],predicted_matrix[i,:,:])))
            else:
                hh_matrix.append(np.linalg.inv(predicted_matrix[i,:,:]))
            #hh_matrix.append(predicted_matrix[i,:,:])

        #return tf.linalg.inv(predicted_matrix+0.0001)
        return np.asarray(hh_matrix)

    def calculate_feature_map(self, input_tensor):
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

    def get_initial_matrix(self, input_img, template_img, initial_type='multi_net'):

        if initial_type=='multi_net':
            input_img_grey=tf.image.rgb_to_grayscale(input_img)
            template_img_new=tf.image.pad_to_bounding_box(template_img, 32, 32, 192, 192)
            template_img_grey=tf.image.rgb_to_grayscale(template_img_new)

            network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
            homography_vector_one=self.regression_network_one.call(network_input,training=False)
            matrix_one=self.construct_matrix_regression(1,homography_vector_one)

            template_img_new=self.LK_layer_regression.projective_inverse_warp(tf.dtypes.cast(template_img,tf.float32), matrix_one)
            template_img_grey=tf.image.rgb_to_grayscale(template_img_new)
            network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
            homography_vector_two=self.regression_network_two.call(network_input,training=False)
            matrix_two=self.construct_matrix_regression(1,homography_vector_one,homography_vector_two)

            template_img_new=self.LK_layer_regression.projective_inverse_warp(tf.dtypes.cast(template_img,tf.float32), matrix_two)
            template_img_grey=tf.image.rgb_to_grayscale(template_img_new)
            network_input=tf.concat([template_img_grey,input_img_grey],axis=-1)
            homography_vector_three=self.regression_network_three.call(network_input,training=False)
            matrix_three=None

            extra=tf.ones((1,1))
            initial_matrix=tf.concat([homography_vector_three,extra],axis=-1)
            initial_matrix=tf.reshape(initial_matrix,[1,3,3])
            initial_matrix=np.dot(initial_matrix[0,:,:], np.linalg.inv(matrix_two[0,:,:]))
            matrix_three=initial_matrix=np.expand_dims(initial_matrix,axis=0)

        return initial_matrix

    def get_dlk_feature_map(self, input_img, template_img, feature_map_type='special'):

        input_feature_one=self.level_one_input.call(input_img,training=False)
        template_feature_one=self.level_one_template.call(template_img,training=False)

        input_feature_two=self.level_two_input.call(input_feature_one,training=False)
        template_feature_two=self.level_two_template.call(template_feature_one,training=False)

        input_feature_three=self.level_three_input.call(input_feature_two,training=False)
        template_feature_three=self.level_three_template.call(template_feature_two,training=False)

        if feature_map_type=='special':

            input_feature_map_one=self.calculate_feature_map(input_feature_one)
            template_feature_map_one=self.calculate_feature_map(template_feature_one)

            input_feature_map_two=self.calculate_feature_map(input_feature_two)
            template_feature_map_two=self.calculate_feature_map(template_feature_two)

            input_feature_map_three=self.calculate_feature_map(input_feature_three)
            template_feature_map_three=self.calculate_feature_map(template_feature_three)

        return (input_feature_map_one, template_feature_map_one), \
                (input_feature_map_two, template_feature_map_two), \
                (input_feature_map_three, template_feature_map_three)

    def dlk_estimation(self, initial_matrix, feature_maps, u_list, v_list):
        input_feature_map_one, template_feature_map_one = feature_maps[0]
        input_feature_map_two, template_feature_map_two = feature_maps[1]
        input_feature_map_three, template_feature_map_three = feature_maps[2]
        fk_loop=self.input_parameters.num_iters

        cornner_error_previous=0.0
        updated_matrix=initial_matrix
        for j in range(fk_loop):
            try:
                updated_matrix=self.LK_layer_three.update_matrix(template_feature_map_three,input_feature_map_three,updated_matrix)
                updated_matrix_back=self.construct_matrix(updated_matrix,scale_factor=4.0,batch_size=1)
                cornner_error, _, _=self.average_cornner_error(1,updated_matrix_back,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
                if np.abs(cornner_error-cornner_error_previous)<1.0:
                    break
                cornner_error_previous=cornner_error

            except Exception as e:
                print (e)

        cornner_error_previous=0.0
        updated_matrix=self.construct_matrix(updated_matrix,scale_factor=2.0,batch_size=1)
        for j in range(fk_loop):
            try:
                updated_matrix=self.LK_layer_two.update_matrix(template_feature_map_two,input_feature_map_two,updated_matrix)
                updated_matrix_back=self.construct_matrix(updated_matrix,scale_factor=2.0,batch_size=1)
                cornner_error, _, _=self.average_cornner_error(1,updated_matrix_back,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
                if np.abs(cornner_error-cornner_error_previous)<0.1:
                    break
                cornner_error_previous=cornner_error
            except Exception as e:
                print (e)

        cornner_error_previous=0.0
        updated_matrix=self.construct_matrix(updated_matrix,scale_factor=2.0,batch_size=1)
        for j in range(fk_loop):
            try:
                updated_matrix=self.LK_layer_one.update_matrix(template_feature_map_one,input_feature_map_one,updated_matrix)
                cornner_error, _, _=self.average_cornner_error(1,updated_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
                if np.abs(cornner_error-cornner_error_previous)<0.01:
                    break
                cornner_error_previous=cornner_error
            except Exception as e:
                print (e)

        return updated_matrix

    def draw_box(self, input_img, u_list, v_list, color=(0, 255, 0), thickness=2):
        pt1 = (int(u_list[0]), int(v_list[0]))
        pt2 = (int(u_list[1]), int(v_list[1]))
        pt3 = (int(u_list[2]), int(v_list[2]))
        pt4 = (int(u_list[3]), int(v_list[3]))

        input_img = cv2.line(input_img, pt1, pt2, color, thickness=thickness)
        input_img = cv2.line(input_img, pt2, pt3, color, thickness=thickness)
        input_img = cv2.line(input_img, pt3, pt4, color, thickness=thickness)
        input_img = cv2.line(input_img, pt4, pt1, color, thickness=thickness)

        return input_img

    def test(self):

        for iters in range(5):
            input_img,u_list,v_list,template_img=self.data_loader_caller.data_read_batch(batch_size=1)

            if len(np.shape(input_img))<2:
                break

            template_img_disp=tf.image.pad_to_bounding_box(template_img, 32, 32, 192, 192)
            cv.imwrite(f'out/template_img{iters}.png', template_img_disp[0].numpy() * 255)

            input_img_disp = input_img[0] * 255
            # gt box
            input_img_disp = self.draw_box(input_img_disp.copy(), u_list[0], v_list[0], color=(0, 0, 255), thickness=2)

            initial_matrix = self.get_initial_matrix(input_img, template_img, self.input_parameters.initial_type)
            cornner_error_pre, u_predict_pre, v_predict_pre=self.average_cornner_error(1,initial_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
            initial_matrix=self.construct_matrix(initial_matrix,scale_factor=0.25,batch_size=1)

            # est box
            input_img_disp_pre = self.draw_box(input_img_disp.copy(), u_predict_pre[0], v_predict_pre[0], color=(0, 255, 255), thickness=2)
            cv.imwrite(f'out/input_img_disp_pre_{iters}.png', input_img_disp_pre)

            feature_maps_one, feature_maps_two, feature_maps_three = self.get_dlk_feature_map(input_img, template_img, self.input_parameters.feature_map_type)
            cv.imwrite(f'out/input_feature_map_one{iters}.png', feature_maps_one[0][0].numpy() * 255)
            cv.imwrite(f'out/template_feature_map_one{iters}.png', feature_maps_one[1][0].numpy() * 255)

            if self.input_parameters.if_LK:
                feature_maps = [feature_maps_one, feature_maps_two, feature_maps_three]
                predicted_matrix = self.dlk_estimation(initial_matrix, feature_maps, u_list, v_list)
                cornner_error, u_predict, v_predict=self.average_cornner_error(1,predicted_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
                print(f'{iters} pre: {cornner_error_pre}, final: {cornner_error}')

                # est box
                input_img_disp_final = self.draw_box(input_img_disp.copy(), u_predict[0], v_predict[0], color=(0, 255, 255), thickness=2)
                cv.imwrite(f'out/input_img_disp_final_{iters}.png', input_img_disp_final)

                template_img_new=self.LK_layer_regression.projective_inverse_warp(tf.dtypes.cast(template_img,tf.float32), np.linalg.inv(predicted_matrix))
                cv.imwrite(f'out/template_img_reg{iters}.png', template_img_new[0].numpy() * 255)
            print('saved')

    def test_manual(self):

        for iters in range(3):
            input_img,u_list,v_list,template_img=self.data_loader_caller.data_read_batch(batch_size=1)

        template_img_disp=tf.image.pad_to_bounding_box(template_img, 32, 32, 192, 192)
        cv.imwrite(f'out/template_img_disp.png', template_img_disp[0].numpy() * 255)
        cv.imwrite(f'out/input_img_disp.png', input_img[0] * 255)

        # template img 128x128x3 -> initial_guess_img 192x192x3
        initial_matrix = self.get_initial_matrix(input_img, template_img, self.input_parameters.initial_type)
        template_img_new=self.LK_layer_regression.projective_inverse_warp(tf.dtypes.cast(template_img,tf.float32), np.linalg.inv(initial_matrix))
        cv.imwrite(f'out/before.png', template_img_new[0].numpy()*255.)

        # initial_guess_img 192x192x3 -> template img 128x128x3
        LK_layer = Lucas_Kanade_layer(batch_size=1,height_template=128,width_template=128,num_channels=3)
        template_img_new=LK_layer.projective_inverse_warp(tf.dtypes.cast(template_img_new,tf.float32), initial_matrix)
        cv.imwrite(f'out/test.png', template_img_new[0].numpy()*255.)

        cornner_error_pre, u_predict_pre, v_predict_pre=self.average_cornner_error(1,initial_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)
        initial_matrix=self.construct_matrix(initial_matrix,scale_factor=0.25,batch_size=1)

        feature_maps_one, feature_maps_two, feature_maps_three = self.get_dlk_feature_map(input_img, template_img, self.input_parameters.feature_map_type)

        cv.imwrite(f'out/input_feature_map_one.png', feature_maps_one[0][0].numpy() * 255)
        cv.imwrite(f'out/template_feature_map_one.png', feature_maps_one[1][0].numpy() * 255)

        feature_maps = [feature_maps_one, feature_maps_two, feature_maps_three]
        predicted_matrix = self.dlk_estimation(initial_matrix, feature_maps, u_list, v_list)
        cornner_error, u_predict, v_predict=self.average_cornner_error(1,predicted_matrix,u_list,v_list,top_left_u=0,top_left_v=0,bottom_right_u=127,bottom_right_v=127)

        template_img_new=self.LK_layer_regression.projective_inverse_warp(tf.dtypes.cast(template_img,tf.float32), np.linalg.inv(predicted_matrix))
        cv.imwrite(f'out/after.png', template_img_new[0].numpy()*255.)


if __name__ == '__main__':
    # set random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    np.set_printoptions(suppress=True)

    input_parameters = parse_args()
    model = DLK_Homography(input_parameters)
    #model.test()
    model.test_manual()
