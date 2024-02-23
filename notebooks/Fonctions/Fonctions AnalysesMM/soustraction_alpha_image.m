function [MM_output,valeur_alpha] = soustraction_alpha_image(MM_input,option_soustraction)
%% === Descriptif ===
%soustraction_alpha_image - Fonction permettant de soustraire à une image de 
% Mueller la quantité alpha*Identité

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    MM_input - Image de la matrice de Mueller d'entrée
%    option_soustraction -Chaine de caractères.  permet de sélectionner le 
%    type de alpha à soustraire (alpha_max ou aure type de alpha)
%
% Outputs:
%    MM_output : Image de la matrice de Mueller après soustraction
%    valeur_alpha : Image de la valeur de alpha soustraire sur la matrice
%    MM_input


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% 2017-11-18; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------

[sizeL,sizeC,~]=size(MM_input);

if strcmp(option_soustraction, 'alpha_max')
    
    for indL=1:sizeL
        
        for indC=1:sizeC
            
            M=(reshape(MM_input(indL,indC,:),4,4));
            M(1,1)=1;
            
            [alpha_max]=limite_speculaire(M);
            
            M=M-(alpha_max-0.01)*eye(4);
            M=M./M(1,1);
            
            MM_output(indL,indC,:)=reshape(M,1,1,16);
            valeur_alpha(indL,indC)=alpha_max;
            
        end
    end
    
elseif strcmp(option_soustraction, 'alpha_0')
    
    for indL=1:sizeL
        
        for indC=1:sizeC
            
            M=(reshape(MM_input(indL,indC,:),4,4));
%           M(1,1)=1;
%           M=M*0.25;
            vp_muel=eig(M);
            vp_muel_sorted=sort(vp_muel);
            alpha_0=real(vp_muel_sorted(1));
            
            alpha_0_final=alpha_0-0.01;
            
            M=M-(alpha_0_final)*eye(4);
            valeur_alpha(indL,indC)=alpha_0_final;
            M=M./M(1,1);
            
            MM_output(indL,indC,:)=reshape(M,1,1,16);

            
        end
    end
    
end

end

%% ------------- END OF CODE --------------

