function [MRL,MRC,total_retardance,retardance_vector,linear_retardance,circular_retardance,orientation_linear_retardance,orientation_linear_retardance_full]=extraction_parametres_matrice_retard(MR,choix_sous_decomposition)
%% === Descriptif ===
%extraction_parametres_matrice_retard - permet d'extraire les
%paramètres polarimétriques du retard à partir d'une matrice de retard MR

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    MR - Matrice de Mueller de retard
%    choix_sous_decomposition - Chaine de caractères. Si 'LIN-CIR' est
%    choisie, alors la matrice de retard sera décomposée selon la forme
%    d'un produit d'un déphaseur linéaire par un déphaseur circulaire. Si
%    'CIR-LIN' est choisie, c'est l'inverse.
%
% Outputs:
%    MRL - Matrice du déphaseur linéaire
%    MRC - Matrice du déphaseur circulaire
%    total_retardance - Retard de phase total. Correspond au RetS
%    retardance_vector - Vecteur de retard de phase
%    linear_retardance - Retard de phase linéaire
%    circular_retardance - Retard de phase circulaire
%    orientation_linear_retardance - Orientation des axes propres du retard
%    de phase linéaire (donné entre 0° et 90°)
%    orientation_linear_retardance_full - Orientation des axes propres du 
%    retard de phase linéaire (donné entre 0° et 180°). Vrai à condition
%    que le retard de phase linéaire n'excède pas 180°


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------


argu= 0.5*(MR(2,2)+MR(3,3)+MR(4,4))-0.5;

if abs(argu)>1   
    if argu>0    
        R=acos(1);     
    else    
        R=acos(-1);     
    end 
else  
    R=acos(real(argu));                                                     % [C6]
end

total_retardance=R*180/pi;
indice_normalisation_retard=1/(2*sin(R));

% Composantes du vecteur retard

a1=R*indice_normalisation_retard*(MR(3,4)-MR(4,3));
a2=R*indice_normalisation_retard*(MR(4,2)-MR(2,4));
a3=R*indice_normalisation_retard*(MR(2,3)-MR(3,2));
retardance_vector=[1,a1,a2,a3]';                                             % [C7]

% Extraction des retards de phase linéaire et circulaire

linear_retardance=acosd(real(MR(4,4)));                                      % [C8]
circular_retardance=atand(real((MR(3,2)-MR(2,3))/(MR(3,3)+MR(2,2))));        % [C9]


%% Décomposition de la matrice retard en un produit linéaire - circulaire

MRC=rota(circular_retardance/2);

if abs(circular_retardance)<0.0001
    MRL=MR;
elseif strcmp(choix_sous_decomposition, 'LIN-CIR')
    MRL=MR*MRC.';
    circular_retardance_temp=atand(real((MRL(3,2)-MRL(2,3))/(MRL(3,3)+MRL(2,2))));
    if abs(circular_retardance_temp)>abs(circular_retardance)
        MRL=MR*MRC;
    end
elseif strcmp(choix_sous_decomposition, 'CIR-LIN')
    MRL=MRC.'*MR;
    circular_retardance_temp=atand(real((MRL(3,2)-MRL(2,3))/(MRL(3,3)+MRL(2,2))));
    if abs(circular_retardance_temp)>abs(circular_retardance)
        MRL=MRC*MR;
    end
    
end


%% Détermination de l'azimut du retard de phase linéaire entre 0° et 90° : orientation_linear_retardance

if total_retardance<0
 %ORN    orientation_linear_retardance=0;
  %ORN   orientation_linear_retardance_full=0; 
else  
   %ORN  orientation_linear_retardance = 0.5*atand(real(MRL(2,4)/MRL(4,3)));        % [C10]
      orientation_linear_retardance = atan2d(MRL(2,4),MRL(4,3));   %ORN
   if sign(orientation_linear_retardance)<0  
  %ORN      orientation_linear_retardance = 90-abs(orientation_linear_retardance);
  orientation_linear_retardance = 360-abs(orientation_linear_retardance);  %ORN 
    else
     %ORN    orientation_linear_retardance = orientation_linear_retardance;
    end
    orientation_linear_retardance_full = 0.5*orientation_linear_retardance; %ORN 
    
    
       orientation_linear_retardance=90+orientation_linear_retardance_full;     %ORN 
    %% Détermination de l'azimut du retard de phase linéaire entre 0° et 180° : orientation_linear_retardance_full (n'a de sens que si le retard de phase linéaire de la matrice n'excède pas pi)
        if sign(MRL(2,4))<0                                                       % [C11]
     %ORN   orientation_linear_retardance_full=orientation_linear_retardance;  
    else      
    %ORN    orientation_linear_retardance_full=90+orientation_linear_retardance;  
    end
end

end

%% ------------- END OF CODE --------------


% [C6] : Retard de phase total. Voir équation (17) de la référence [1]
% [C7] : Vecteur retard de la matrice du déphaseur. Voir équation (17) de la référence [1]
% [C8] : Extraction du retard de phase linéaire de la matrice du déphaseur. Le retard de  phase linéaire d'un élément est toujours localisé sur le terme MR44,
%        même en présence de retard de phase circulaire. Voir équations 6.22, 6.23 et 6.24 de la thèse de Jérémy Vizet
% [C9] : Extraction du retard de phase circulaire de la matrice MR. Même en présence de retard de phase linéaire, le retard de phase circulaire peut toujours s'obtenir par 
%        cette formule. Voir équations 6.22, 6.23 et 6.24 de la thèse de Jérémy Vizet. Voir équation (13) de la référence (4). Attention : le retard de phase circulaire 
%        vaut deux fois l'angle de rotation (optical rotation) induit sur les états de polarisation par l'activité optique.
% [C10] : Extraction de l'orientation des axes neutres de la biréfringence linéaire dans le cas où la biréfringence circulaire est négligeable. Pour obtenir cette formule,
%         il suffit d'écrire la forme d'un déphaseur linéaire orienté selon un angle theta dans un plan perpendiculaire à la direction du faisceau de lumière incident. 
%         L'orientation theta peut alors être déterminée grâce aux termes MR24 et MR43. Voir équation 6.68 de la thèse de Jérémy Vizet.
% [C11] : Il est possible de déterminer l'orientation des axes neutres de la biréfringence linéaire entre 0° et 180° en observant le signe du terme MR24 de la matrice du déphaseur en supposant que le
%         retard de phase linéaire n'excède pas 180°. Dans ces conditions, ce terme vaut sin(2*theta)*sin(delta), et un changement de son signe sur une image indique que
%         l'orientation theta a excédé 90°. Il suffit alors de rajouter un "offset" de 90° à la valeur de theta calculée de manière "brute" dans le cas où cela se produit.

