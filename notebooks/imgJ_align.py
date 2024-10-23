import os
import sys
import imagej

def align_imgJ(ij, path_tmp):

    # get the current working directory for inserting correctly the file paths in the macro
    path_tmp_no_start = path_tmp
    path_tmp = f'"{path_tmp}'

    macro = """
        open(""" + path_tmp + """/moving.png");
        open(""" + path_tmp + """/img_x.tif");
        call("bunwarpj.bUnwarpJ_.loadLandmarks", """ + path_tmp + """/coordinates.txt");
        run("bUnwarpJ", "load=""" + path_tmp_no_start + """/coordinates.txt source_image=img_x.tif target_image=moving.png registration=Accurate image_subsample_factor=0 initial_deformation=Coarse final_deformation=[Very Fine] divergence_weight=0.1 curl_weight=0.1 landmark_weight=1.5 image_weight=0 consistency_weight=10 stop_threshold=0.01 save_transformations save_direct_transformation=""" + path_tmp_no_start + """/global_img_direct_transf.txt save_inverse_transformation=""" + path_tmp_no_start + """/img_x_inverse_transf.txt");
        close();
        saveAs("Tiff", """ + path_tmp + """/registered_img_x.tif");
        close();
        close();
        close();
        open(""" + path_tmp + """/moving.png");
        open(""" + path_tmp + """/img_y.tif");
        call("bunwarpj.bUnwarpJ_.loadLandmarks", """ + path_tmp + """/coordinates.txt");
        run("bUnwarpJ", "load=""" + path_tmp_no_start + """/coordinates.txt source_image=img_y.tif target_image=moving.png registration=Accurate image_subsample_factor=0 initial_deformation=Coarse final_deformation=[Very Fine] divergence_weight=0.1 curl_weight=0.1 landmark_weight=1.5 image_weight=0 consistency_weight=10 stop_threshold=0.01 save_transformations save_direct_transformation=""" + path_tmp_no_start + """/global_img_direct_transf.txt save_inverse_transformation=""" + path_tmp_no_start + """/img_y_inverse_transf.txt");
        close();
        saveAs("Tiff", """ + path_tmp + """/registered_img_y.tif");
        close();
        close();
        close();
        """
    ij.py.run_macro(macro)
    
def align_imgJ(ij, path_tmp):

    # get the current working directory for inserting correctly the file paths in the macro
    path_tmp_no_start = path_tmp
    path_tmp = f'"{path_tmp}'

    macro = """
        open(""" + path_tmp + """/img_x.tif");
        open(""" + path_tmp + """/img_x.tif");
        call("bunwarpj.bUnwarpJ_.loadLandmarks", """ + path_tmp + """/coordinates.txt");
        run("bUnwarpJ", "load=""" + path_tmp_no_start + """/coordinates.txt source_image=img_x.tif target_image=img_x.tif registration=Accurate image_subsample_factor=0 initial_deformation=Coarse final_deformation=[Very Fine] divergence_weight=0.1 curl_weight=0.1 landmark_weight=1.5 image_weight=0 consistency_weight=10 stop_threshold=0.01 save_transformations save_direct_transformation=""" + path_tmp_no_start + """/global_img_direct_transf.txt save_inverse_transformation=""" + path_tmp_no_start + """/img_x_inverse_transf.txt");
        close();
        saveAs("Tiff", """ + path_tmp + """/registered_img_x.tif");
        close();
        close();
        close();
        open(""" + path_tmp + """/img_y.tif");
        open(""" + path_tmp + """/img_y.tif");
        call("bunwarpj.bUnwarpJ_.loadLandmarks", """ + path_tmp + """/coordinates.txt");
        run("bUnwarpJ", "load=""" + path_tmp_no_start + """/coordinates.txt source_image=img_y.tif target_image=img_y.tif registration=Accurate image_subsample_factor=0 initial_deformation=Coarse final_deformation=[Very Fine] divergence_weight=0.1 curl_weight=0.1 landmark_weight=1.5 image_weight=0 consistency_weight=10 stop_threshold=0.01 save_transformations save_direct_transformation=""" + path_tmp_no_start + """/global_img_direct_transf.txt save_inverse_transformation=""" + path_tmp_no_start + """/img_y_inverse_transf.txt");
        close();
        saveAs("Tiff", """ + path_tmp + """/registered_img_y.tif");
        close();
        close();
        close();
        """
    ij.py.run_macro(macro)
    
def main():
    """
    main function to perform the alignment of the histology images with the polarimetry images

    Parameters
    ----------
    
    Returns
    -------
    """

    path_tmp = os.path.abspath(sys.argv[1]).replace('\\','/')
    ij = imagej.init(os.path.join(os.path.dirname(os.path.realpath(__file__)).split('src')[0], 'Fiji.app'), mode='interactive')
    align_imgJ(ij, path_tmp)
    
if __name__ == "__main__":
    main()
    
    
