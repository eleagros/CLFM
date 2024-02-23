function[MD,mD,Mdelta,MR,M11,total_diattenuation,linear_diattenuation,circular_diattenuation,total_depolarization,total_retardance,vecteur_retard,linear_retardance,orientation_linear_retardance,orientation_linear_retardance_full,circular_retardance]=polardecomposition0_paper(muel)


format short




I=[1 0 0;
    0 1 0;
    0 0 1];

M11=muel(1,1);

%muel=muel./muel(1,1);

muel(1,1)=1;

%% =================  Extraction de la matrice de diatténuation  ================= %%


pvec=[muel(2,1),muel(3,1),muel(4,1)]*(1/muel(1,1));

dvec=[muel(1,2),muel(1,3),muel(1,4)]*(1/muel(1,1));


%% Extraction des paramètres de diatténuation %%

D=((muel(1,2)^2+muel(1,3)^2+muel(1,4)^2)^0.5);


m=muel(2:4,2:4);


D1=(1-D^2)^0.5

total_diattenuation=D;
linear_diattenuation=sqrt(muel(1,2)^2+muel(1,3)^2);
circular_diattenuation=abs(muel(1,4));


if D==0
    
    muel_0=muel;
    
    MD=eye(4);
    
else
    
    mD=D1*I+(1-D1)*(dvec')*dvec/D^2;
    
    MD=[1,dvec;
        dvec',mD];
    MD;
    muel_0=muel/MD;
    
end









%% =================  Extraction de la matrice de dépolarisation  ================= %%


m_1=muel_0(2:4,2:4);
 
l_0=eig(m_1*m_1');

m_0=inv(m_1*m_1'+((l_0(1)*l_0(2))^0.5+(l_0(2)*l_0(3))^0.5+(l_0(3)*l_0(1))^0.5)*I);

m_00=(l_0(1)^0.5+l_0(2)^0.5+l_0(3)^0.5)*m_1*(m_1')+I*(l_0(1)*l_0(2)*l_0(3))^0.5;

% if det(m_1)>=0
%     
%     mdelta=m_0*m_00;
%     
% else
%     
%     mdelta=-m_0*m_00;
%     
% end

mdelta=m_0*m_00;

total_depolarization=1-(abs(mdelta(1,1))+abs(mdelta(2,2))+abs(mdelta(3,3)))/3;

nul=(pvec'-m*dvec')/D1^2;

Mdelta=[1 0 0 0;
    nul mdelta];

Mdelta;








%% =================  Extraction de la matrice du déphaseur  ================= %%

MR=Mdelta\muel_0;

trmR=(MR(2,2)+MR(3,3)+MR(4,4))/2;

argu=trmR-1/2;



%% Extraction des paramètres du retard de phase (R : retard de phase total) %%

if abs(argu)>1
    
    if argu>0
        
        R=acos(1);
    else
        
        R=acos(-1);
        
    end
    
else
    
    R=acos(argu);
    
end

total_retardance=R*180/pi;


indice_normalisation_retard=1/(2*sin(R));

a1=indice_normalisation_retard*(MR(3,4)-MR(4,3));

a2=indice_normalisation_retard*(MR(4,2)-MR(2,4));

a3=indice_normalisation_retard*(MR(2,3)-MR(3,2));



vecteur_retard=[1,a1,a2,a3]';
linear_retardance=acosd(real(MR(4,4)));
circular_retardance=atand(real((MR(3,2)-MR(2,3))/(MR(3,3)+MR(2,2))));





%% Détermination de l'azimut du retard de phase linéaire entre 0° et 90° : orientation_linear_retardance


orientation_linear_retardance = 0.5*atand(real(MR(2,4)/MR(4,3)));

if sign(orientation_linear_retardance)<0
    
    orientation_linear_retardance = 90-abs(orientation_linear_retardance);
else
    orientation_linear_retardance = orientation_linear_retardance;
end

%% Détermination de l'azimut du retard de phase linéaire entre 0° et 180° : orientation_linear_retardance_full (n'a de sens que si le retard de phase linéaire de la matrice n'excède pas pi)

check=rota(-(orientation_linear_retardance))*MR*rota(orientation_linear_retardance);

orientation_linear_retardance_full=orientation_linear_retardance;

if check(3,4)<0
    
    orientation_linear_retardance_full=90+orientation_linear_retardance_full;
    
end









return