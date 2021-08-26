use the matlab script 'gen_akagi_data.mlx' to generate input & template image for DLK method. for sample data refer 'sample_dlk_data.zip' -> contains image pairs with gt homography control points and 'Sample_AkagiEarth_DS.zip' -> DLK dataset

Following are the steps involved in matlab script
1. after pre-processing(if required) satellite and drone image, using cpselect tool fixed points (fp) in satellite image and moving points (mp) in drone image are manually selected to find the projective homography. The better these control points, the better is the estimate. [refer](https://de.mathworks.com/help/images/registering-an-aerial-photo-to-an-orthophoto.html)
2. using the above homography we find the corresponding points of satellite image(tgt_points) [[32 32];[159 32];[32 159]; [159 159]] in drone image (src_points)-> these corner points are fixed as per DLK paper.
3. the computed src_points will be used as supervised label for DLK method.
4. 
