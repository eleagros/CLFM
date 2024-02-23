function[Image_MD1,Image_MD2,...
    Image_MR,Image_MRL,Image_MRC,Image_MR1,Image_MRL1,Image_MRC1,Image_MR2,Image_MRL2,Image_MRC2,Image_matrice_check,...
    Image_Mdelta,...
    Image_total_diattenuation_1,Image_linear_diattenuation_1,Image_orientation_linear_diattenuation_1,Image_circular_diattenuation_1,...
    Image_total_diattenuation_2,Image_linear_diattenuation_2,Image_orientation_linear_diattenuation_2,Image_circular_diattenuation_2,...
    Image_total_retardance,Image_retardance_vector,Image_linear_retardance,Image_orientation_linear_retardance,Image_orientation_linear_retardance_full,Image_circular_retardance,...
    Image_total_retardance_1,Image_retardance_vector_1,Image_linear_retardance_1,Image_orientation_linear_retardance_1,Image_orientation_linear_retardance_full_1,Image_circular_retardance_1,...
    Image_total_retardance_2,Image_retardance_vector_2,Image_linear_retardance_2,Image_orientation_linear_retardance_2,Image_orientation_linear_retardance_full_2,Image_circular_retardance_2,...
    Image_total_depolarization]=symetric_decomposition_Mueller_image(MM)

%% === Descriptif ===
% symetric_decomposition_Mueller_image : permet de décomposer une image de
% Mueller MM par la décomposition symétrique

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    MM - Matrice de Mueller sous forme d'image

% Outputs:Les indices 1 et 2 qui suivent chacune des variables suivantes
% désignent l'entrée et la sortie dans la décomposition symétrique.
%    Image_MD - Image de la Matrice de diatténuation
%    Image_MR - Image de la Matrice du déphaseur
%    Image_MRL - Image de la Matrice du déphaseur linéaire extrait à partir de MR
%    Image_MRC - Image de la Matrice du déphaseur circulaire extrait à partir de MR
%    Image_Mdelta - Image de la Matrice du dépolariseur
%    Image_total_diattenuation - Image de la Diatténuation totale
%    Image_linear_diattenuation - Image de la Diatténuation linéaire
%    Image_orientation linear_diattenuation - Image de l'orientation des axes propres de la
%    Image_diatténuation linéaire
%    Image_circular_diattenuation - Image de la Diatténuation circulaire
%    Image_total_retardance - Image du Retard de phase total (correspond au RetS) extrait
%    à partir de MR
%    Image_retardance_vector - Image du vecteur de retard de phase extrait à partir de MR
%    Image_linear_retardance - Image du Retard de phase linéaire extrait à partir de MRL
%    Image_orientation_linear_retardance - Image de l'orientation des axes propres du retard
%    de phase linéaire (donné entre 0° et 90°)
%    Image_orientation_linear_retardance_full - Image de l'Orientation des axes propres du 
%    retard de phase linéaire (donné entre 0° et 180°). Vrai à condition
%    que le retard de phase linéaire n'excède pas 180°
%    Image_circular_retardance - Image du Retard de phase circulaire
%    Image_total_depolarization - Image de la Dépolarisation totale calculée à partir de
%    Mdelta

%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017; Last revision: 2017-02-17



%% ------------- BEGIN CODE --------------


[sizeL,sizeC,sizeD]=size(MM);

%% =================  Initialisation des tableaux  ================= %%



Image_MD1=zeros(sizeL,sizeC,sizeD);
Image_MD2=zeros(sizeL,sizeC,sizeD);
Image_MR=zeros(sizeL,sizeC,sizeD);
Image_MRL=zeros(sizeL,sizeC,sizeD);
Image_MRC=zeros(sizeL,sizeC,sizeD);
Image_MR1=zeros(sizeL,sizeC,sizeD);
Image_MRL1=zeros(sizeL,sizeC,sizeD);
Image_MRC1=zeros(sizeL,sizeC,sizeD);
Image_MR2=zeros(sizeL,sizeC,sizeD);
Image_MRL2=zeros(sizeL,sizeC,sizeD);
Image_MRC2=zeros(sizeL,sizeC,sizeD);
Image_Mdelta=zeros(sizeL,sizeC,sizeD);
Image_matrice_check=zeros(sizeL,sizeC,sizeD);


Image_total_diattenuation_1=zeros(sizeL,sizeC);
Image_total_diattenuation_2=zeros(sizeL,sizeC);
Image_linear_diattenuation_1=zeros(sizeL,sizeC);
Image_linear_diattenuation_2=zeros(sizeL,sizeC);
Image_orientation_linear_diattenuation_1=zeros(sizeL,sizeC);
Image_orientation_linear_diattenuation_2=zeros(sizeL,sizeC);
Image_circular_diattenuation_1=zeros(sizeL,sizeC);
Image_circular_diattenuation_2=zeros(sizeL,sizeC);


Image_total_retardance=zeros(sizeL,sizeC);
Image_total_retardance_1=zeros(sizeL,sizeC);
Image_total_retardance_2=zeros(sizeL,sizeC);
Image_retardance_vector= repmat(reshape(zeros(1,4),1,1,4),sizeL,sizeC);
Image_retardance_vector_1= repmat(reshape(zeros(1,4),1,1,4),sizeL,sizeC);
Image_retardance_vector_2= repmat(reshape(zeros(1,4),1,1,4),sizeL,sizeC);
Image_linear_retardance=zeros(sizeL,sizeC);
Image_linear_retardance_1=zeros(sizeL,sizeC);
Image_linear_retardance_2=zeros(sizeL,sizeC);
Image_orientation_linear_retardance=zeros(sizeL,sizeC);
Image_orientation_linear_retardance_1=zeros(sizeL,sizeC);
Image_orientation_linear_retardance_2=zeros(sizeL,sizeC);
Image_orientation_linear_retardance_full=zeros(sizeL,sizeC);
Image_orientation_linear_retardance_full_1=zeros(sizeL,sizeC);
Image_orientation_linear_retardance_full_2=zeros(sizeL,sizeC);
Image_circular_retardance=zeros(sizeL,sizeC);
Image_circular_retardance_1=zeros(sizeL,sizeC);
Image_circular_retardance_2=zeros(sizeL,sizeC);

Image_total_depolarization=zeros(sizeL,sizeC);


h = waitbar(0,'Please wait...');

for indL=1:sizeL
    
    for indC=1:sizeC
        
        M=(reshape(MM(indL,indC,:),4,4));
        M(1,1)=1;
        
        
        %% Lancement de la décomposition de la matrice de Mueller M
        
       [MD1,MD2,MR,MRL,MRC,MR1,MRL1,MRC1,MR2,MRL2,MRC2,Mdelta,matrice_check,...
        total_diattenuation_1,linear_diattenuation_1,orientation_linear_diattenuation_1,circular_diattenuation_1,...
        total_diattenuation_2,linear_diattenuation_2,orientation_linear_diattenuation_2,circular_diattenuation_2,...
        total_retardance,retardance_vector,linear_retardance,orientation_linear_retardance,orientation_linear_retardance_full,circular_retardance,...
        total_retardance_1,retardance_vector_1,linear_retardance_1,orientation_linear_retardance_1,orientation_linear_retardance_full_1,circular_retardance_1,...
        total_retardance_2,retardance_vector_2,linear_retardance_2,orientation_linear_retardance_2,orientation_linear_retardance_full_2,circular_retardance_2,...
        total_depolarization]=symetric_decomposition_Mueller(M);

        
        
        %% Images des matrices MD (diatténuateur), MR (déphaseur), Mdelta (dépolariseur) %%
        
        Image_MD1(indL,indC,:)=real(reshape(MD1,1,1,16));
        Image_MD2(indL,indC,:)=real(reshape(MD2,1,1,16));
        Image_MR(indL,indC,:)=reshape(MR,1,1,16);
        Image_MRL(indL,indC,:)=reshape(MRL,1,1,16);
        Image_MRC(indL,indC,:)=reshape(MRC,1,1,16);
        Image_MR1(indL,indC,:)=reshape(MR1,1,1,16);
        Image_MRL1(indL,indC,:)=reshape(MRL1,1,1,16);
        Image_MRC1(indL,indC,:)=reshape(MRC1,1,1,16);
        Image_MR2(indL,indC,:)=reshape(MR2,1,1,16);
        Image_MRL2(indL,indC,:)=reshape(MRL2,1,1,16);
        Image_MRC2(indL,indC,:)=reshape(MRC2,1,1,16);
        Image_Mdelta(indL,indC,:)=reshape(Mdelta,1,1,16);
        Image_matrice_check(indL,indC,:)=reshape(matrice_check,1,1,16);
        
        %% Images des paramètres de la diatténuation %%
        
        Image_total_diattenuation_1(indL,indC)=total_diattenuation_1;
        Image_total_diattenuation_2(indL,indC)=total_diattenuation_2;
        Image_linear_diattenuation_1(indL,indC)=linear_diattenuation_1;
        Image_linear_diattenuation_2(indL,indC)=linear_diattenuation_2;
        Image_orientation_linear_diattenuation_1(indL,indC)=orientation_linear_diattenuation_1;
        Image_orientation_linear_diattenuation_2(indL,indC)=orientation_linear_diattenuation_2;
        Image_circular_diattenuation_1(indL,indC)=circular_diattenuation_1;
        Image_circular_diattenuation_2(indL,indC)=circular_diattenuation_2;
        
        %% Images des paramètres du retard de phase %%
        
        Image_total_retardance(indL,indC)=total_retardance;
        Image_total_retardance_1(indL,indC)=total_retardance_1;
        Image_total_retardance_2(indL,indC)=total_retardance_2;
        Image_retardance_vector(indL,indC,:)=retardance_vector;
        Image_retardance_vector_1(indL,indC,:)=retardance_vector_1;
        Image_retardance_vector_2(indL,indC,:)=retardance_vector_2;
        Image_linear_retardance(indL,indC)=linear_retardance;
        Image_linear_retardance_1(indL,indC)=linear_retardance_1;
        Image_linear_retardance_2(indL,indC)=linear_retardance_2;
        Image_orientation_linear_retardance(indL,indC)=orientation_linear_retardance;
        Image_orientation_linear_retardance_1(indL,indC)=orientation_linear_retardance_1;
        Image_orientation_linear_retardance_2(indL,indC)=orientation_linear_retardance_2;
        Image_orientation_linear_retardance_full(indL,indC)=orientation_linear_retardance_full;
        Image_orientation_linear_retardance_full_1(indL,indC)=orientation_linear_retardance_full_1;
        Image_orientation_linear_retardance_full_2(indL,indC)=orientation_linear_retardance_full_2;
        Image_circular_retardance(indL,indC)=circular_retardance;
        Image_circular_retardance_1(indL,indC)=circular_retardance_1;
        Image_circular_retardance_2(indL,indC)=circular_retardance_2;
        
        %% Images du paramètre de dépolarisation totale %%
        
        Image_total_depolarization(indL,indC)=total_depolarization;
        
    end
    
   waitbar(indL/sizeL);
   
end

close(h);

end

%% ------------- END OF CODE --------------

