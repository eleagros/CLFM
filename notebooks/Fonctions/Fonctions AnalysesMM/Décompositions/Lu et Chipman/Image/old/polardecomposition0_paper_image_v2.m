function[ImageMD,ImageMdelta,ImageMR,M11,total_diattenuation,linear_diattenuation,circular_diattenuation,total_depolarization,total_retardance,retardance_vector,linear_retardance,orientation_linear_retardance,orientation_linear_retardance_full,circular_retardance,physical_criterion]=polardecomposition0_paper_image(MM)

format short

[sizeX,sizeY,sizeZ]=size(MM);
Dsign = diag([1; -1; -1; -1]);

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

for indX=1:sizeX
    
    for indY=1:sizeY
        
        
            muel=(reshape(MM(indX,indY,:),4,4));
            M11(indX,indY)=muel(1,1);
           %muel=muel./muel(1,1);
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
            
       
            
            
            physical_criterion(indX,indY)=1;
     
            
            
%% =================  Extraction de la matrice de diatténuation  ================= %%
            
            
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
                
                mD=D1*eye(3)+(1-D1)*(dvec')*dvec/D^2;
                MD=[1,dvec;
                    dvec',mD];
                muel_0=muel/MD;
           
            end
            
            
            ImageMD(indX,indY,:)=real(reshape(MD,1,1,16));
            
            
            
            
            
        
            
    
            
%% =================  Extraction de la matrice du déphaseur  ================= %%



            m_1=muel_0(2:4,2:4);
            [U,S,V]=svd(m_1);
            
            MR=[1,zeros(1,3);
                zeros(1,3)',U*V'];
           
            ImageMR(indX,indY,:)=reshape(MR,1,1,16);
            
            
            
            
            
%% =================  Extraction de la matrice de dépolarisation  ================= %%
            
            Mdelta=muel_0*MR';
          
            if det(MR) < 0
                MR = Dsign * MR;
                Mdelta = Mdelta * Dsign;
            end

%% =================  Extraction des paramètres de la dépolarisation et du retard  ================= %%
   
   
            
            % Extraction des paramètres de la dépolarisation %
            
            total_depolarization(indX,indY)=1-(abs(Mdelta(2,2))+abs(Mdelta(3,3))+abs(Mdelta(4,4)))/3;   
            ImageMdelta(indX,indY,:)=reshape(Mdelta,1,1,16);
      
            
            
            
            % Extraction des paramètres du retard de phase (R : retard de phase total) %
           
            
           
            argu= 0.5*trace(MR(2:4,2:4))-0.5;
            

            
            if abs(argu)>1
                
                if argu>0
                    
                    R=acosd(1);
                else
                    
                    R=acosd(-1);
                    
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
            
            
            
            
%% Détermination de l'azimut du retard de phase linéaire entre 0° et 90° : orientation_linear_retardance
            
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
    
    waitbar(indX/sizeX);
    
end


close(h);


