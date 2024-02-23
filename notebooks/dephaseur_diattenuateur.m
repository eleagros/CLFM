function [ muel ] = dephaseur_diattenuateur (tau, psi , delta, theta)

psi=psi*pi/180;
delta=delta*pi/180;

muel =  tau*rota(theta)*[1,-cos(2*psi),0,0;
                        -cos(2*psi),1,0,0;
                         0,0,sin(2*psi)*cos(delta),sin(2*psi)*sin(delta);
                         0,0,-sin(2*psi)*sin(delta),sin(2*psi)*cos(delta);]*rota(-theta);
    
end