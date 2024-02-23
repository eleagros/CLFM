function image_la_plus_intense = image_saturation(raw_intensity)
%% === Descriptif ===
%image_saturation - permet de donner le numéro de l'image la plus intense
%parmi les 16 mesurées sur un imageur polarimétrique de Mueller

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    raw_intensity - Variable contenant les 16 images d'intensité brutes
%
% Outputs:
%    image_la_plus_intense : numéro de l'image la plus intensite


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017; Last revision: 2017-12-11

%% ------------- BEGIN CODE --------------

for ind=1:16
    
    moyenne_temp(:,ind)=mean2(raw_intensity(:,:,ind));
    [maximum_intensite,image_la_plus_intense]=max(moyenne_temp);
end

end

%% ------------- END OF CODE --------------