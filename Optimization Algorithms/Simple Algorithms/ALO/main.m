% This code is modified by Dr. Behdad Arandian for ALO optimization
%Just run main and enjoy
clear all 
clc

SearchAgents_no=40; % Number of search agents

Function_name='F23'; % Name of the test function that can be from F1 to F23 (Table 1,2,3 in the paper)

Max_iteration=500; % Maximum number of iterations

% Load details of the selected benchmark function
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);

[Best_score,Best_pos,cg_curve]=ALO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);

figure('Position',[100 100 660 290])
%Draw search space
subplot(1,2,1);
func_plot(Function_name);
title('Test function')
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name,'( x_1 , x_2 )'])
grid off

%Draw objective space
subplot(1,2,2);
semilogy(cg_curve,'Color','r')
title('Convergence curve')
xlabel('Iteration');
ylabel('Best score obtained so far');

axis tight
grid off
box on
legend('ALO')

display(['The best solution obtained by ALO is : ', num2str(Best_pos)]);
display(['The best optimal value of the objective funciton found by ALO is : ', num2str(Best_score)]);

        



