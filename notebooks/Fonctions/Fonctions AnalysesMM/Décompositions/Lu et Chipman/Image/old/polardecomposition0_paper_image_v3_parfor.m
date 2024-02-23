function[ImageMD,ImageMdelta,ImageMR,M11,total_diattenuation,linear_diattenuation,circular_diattenuation,total_depolarization,total_retardance,retardance_vector,linear_retardance,orientation_linear_retardance,orientation_linear_retardance_full,circular_retardance,physical_criterion]=polardecomposition0_paper_image_v3_parfor(MM)

format short

[sizeX,sizeY,sizeZ]=size(MM);

%% =================  Initialisation des tableaux  ================= %%


        ImageMD=zeros(sizeX,sizeY,sizeZ);
        ImageMdelta=zeros(sizeX,sizeY,sizeZ);
        ImageMR=zeros(sizeX,sizeY,sizeZ);
        retardance_vector=zeros(sizeX,sizeY,4);
        M11=zeros(sizeX,sizeY);
        total_diattenuation=zeros(sizeX,sizeY);
        linear_diattenuation=zeros(sizeX,sizeY);
        circular_diattenuation=zeros(sizeX,sizeY);
        total_depolarization=zeros(sizeX,sizeY);
        total_retardance=zeros(sizeX,sizeY);
        linear_retardance=zeros(sizeX,sizeY);
        orientation_linear_retardance=zeros(sizeX,sizeY);
        orientation_linear_retardance_full=zeros(sizeX,sizeY);
        circular_retardance=zeros(sizeX,sizeY);
        physical_criterion=zeros(sizeX,sizeY);
        
        

h = waitbar(0,'Please wait...');

sizetotal=sizeX*sizeY;



tabX=linspace(1,sizeX,sizeX);
tabY=linspace(1,sizeY,sizeY);

tabT(1,:)=linspace(1,sizetotal,sizetotal);
tabT(2,:)=reshape(repmat((1:sizeX)',1,sizeY)',sizetotal,1)';
tabT(3,:)=repmat(tabY,1,sizeX);

% A = (1:sizeX)';
% B = repmat(A,1,sizeY)
% B=reshape(B',sizetotal,1)'

% A = (1:sizeX)';
% B = repmat((1:sizeX)',1,sizeY)
% B=reshape(repmat((1:sizeX)',1,sizeY)',sizetotal,1)'

for INBOUCLE=1:1:sizetotal
            
            indX=tabT(2,INBOUCLE);
            indY=tabT(3,INBOUCLE);
            
            muel=(reshape(MM(indX,indY,:),4,4));
            M11(indX,indY)=muel(1,1);
           muel=muel./muel(1,1);
            muel(1,1)=1;
            
           
           
        % Si la matrice contient des NaN, des Inf, ou des matrices de
        % Mueller non physiques, les matrices de dépolarisation, de
        % diattéuation et de retard sont égales à l'identité. Les autres
        % paramètres sont mis à 0
%         
%         physicity_check=physical_crit(muel)
%          || physicity_check <1
%          || any(any(muel(:,:)>1.1)) 

        if any(any(isnan(muel))) || any(any(isinf(muel))) 
            
            MD = eye(4);
            Mdelta = eye(4);
            MR = eye(4);
            ImageMD(indX,indY,:)=reshape(MD',1,1,16);
            ImageMdelta(indX,indY,:)=reshape(Mdelta',1,1,16);
            ImageMR(indX,indY,:)=reshape(MR',1,1,16);
            retardance_vector(indX,indY,:)=[1,0,0,0]';
            M11(indX,indY)=0;
            total_diattenuation(indX,indY)=0;
            linear_diattenuation(indX,indY)=0;
            circular_diattenuation(indX,indY)=0;
            total_depolarization(indX,indY)=0;
            total_retardance(indX,indY)=0;
            linear_retardance(indX,indY)=0;
            orientation_linear_retardance(indX,indY)=0;
            orientation_linear_retardance_full(indX,indY)=0;
            circular_retardance(indX,indY)=0;
            physical_criterion(indX,indY)=0;
            
           
        else
            
       
            
%             
            physical_criterion(indX,indY)=1;
     
            
            
%% =================  Extraction de la matrice de diatténuation  ================= %%
            
            pvec=[muel(2,1),muel(3,1),muel(4,1)]*(1/muel(1,1));
            dvec=[muel(1,2),muel(1,3),muel(1,4)]*(1/muel(1,1));
            
            % Extraction des paramètres de diatténuation %
            
            D=((muel(1,2)^2+muel(1,3)^2+muel(1,4)^2)^0.5);
            m=muel(2:4,2:4);
            D1=(1-D^2)^0.5;
            total_diattenuation(indX,indY)=D;
             linear_diattenuation(indX,indY)=sqrt(muel(1,2)^2+muel(1,3)^2);
             circular_diattenuation(indX,indY)=abs(muel(1,4));
            
            
            if D==0
                
                muel_0=muel;
                MD=eye(4);
                
            else
                
                
                MD=[1,dvec;
                    dvec',D1*eye(3)+(1-D1)*(dvec')*dvec/D^2];
                muel_0=muel/MD;
           
            end
            
            
             ImageMD(indX,indY,:)=real(reshape(MD,1,1,16));
            
            
            
            
            
            
%% =================  Extraction de la matrice de dépolarisation  ================= %%
            
            
            m_1=muel_0(2:4,2:4);
            l_0=eig(m_1*m_1');
            m_0=eye(3,3)/(m_1*m_1'+((l_0(1)*l_0(2))^0.5+(l_0(2)*l_0(3))^0.5+(l_0(3)*l_0(1))^0.5)*eye(3));
            m_00=(l_0(1)^0.5+l_0(2)^0.5+l_0(3)^0.5)*m_1*(m_1')+eye(3)*(l_0(1)*l_0(2)*l_0(3))^0.5;
            
            if det(m_1)>=0
                
                mdelta=m_0*m_00;
                
            else
                
                mdelta=-m_0*m_00;
                
            end
            
            pdeltavec=(pvec'-m*dvec')/D1^2;
            
            Mdelta=[1 zeros(1,3);
                pdeltavec mdelta];
            
            Mdelta=real(Mdelta);

            
            % Extraction des paramètres de la dépolarisation %
            
             total_depolarization(indX,indY)=1-(abs(mdelta(1,1))+abs(mdelta(2,2))+abs(mdelta(3,3)))/3;   
             ImageMdelta(indX,indY,:)=reshape(Mdelta,1,1,16);
      
            
    
            
%% =================  Extraction de la matrice du déphaseur  ================= %%
            
            MR=real(Mdelta\muel_0);
            ImageMR(indX,indY,:)=reshape(MR,1,1,16);
    
            
            
            % Extraction des paramètres du retard de phase (R : retard de phase total) %
            
             argu= 0.5*(MR(2,2)+MR(3,3)+MR(4,4))-0.5;
            
            
            if abs(argu)>1
                
                if argu>0
                    
                    R=acos(1);
                else
                    
                    R=acos(-1);
                    
                end
                
            else
                
                R=acosd(real(argu));
                
            end
            
            total_retardance(indX,indY)=R;
            indice_normalisation_retard=1/(2*sin(R));
            
            % Composantes du vecteur retard
            
            a1=indice_normalisation_retard*(MR(3,4)-MR(4,3));
            a2=indice_normalisation_retard*(MR(4,2)-MR(2,4));
            a3=indice_normalisation_retard*(MR(2,3)-MR(3,2));
           retardance_vector(indX,indY,:)=[1,a1,a2,a3]';
          
            
            % Extraction des retards de phase linéaire et circulaire
            
            linear_retardance(indX,indY)=acosd(real(MR(4,4)));
            circular_retardance(indX,indY)=atand(real((MR(3,2)-MR(2,3))/(MR(3,3)+MR(2,2))));
            
            
%             
%             
% %% Détermination de l'azimut du retard de phase linéaire entre 0° et 90° : orientation_linear_retardance
%             
            if total_retardance(indX,indY)<0
                orientation_linear_retardance(indX,indY)=0;
                orientation_linear_retardance_full(indX,indY)=0;
                
            else
                
                orientation_linear_retardance(indX,indY) = 0.5*atand(real(MR(2,4)/MR(4,3)));
                
                if sign(orientation_linear_retardance(indX,indY))<0
                    
                    orientation_linear_retardance(indX,indY) = 90-abs(orientation_linear_retardance(indX,indY));
                else
                    orientation_linear_retardance(indX,indY) = orientation_linear_retardance(indX,indY);
                end
                
                
%% Détermination de l'azimut du retard de phase linéaire entre 0° et 180° : orientation_linear_retardance_full (n'a de sens que si le retard de phase linéaire de la matrice n'excède pas pi)
                
                check=rota(-(orientation_linear_retardance(indX,indY)))*MR*rota(orientation_linear_retardance(indX,indY));
                
                if check(3,4)>0
                    
                    orientation_linear_retardance_full(indX,indY)=orientation_linear_retardance(indX,indY);
                    
                else
                    
                    orientation_linear_retardance_full(indX,indY)=90+orientation_linear_retardance(indX,indY);
                    
                end
                
            end
        end
       
 
    end
    



