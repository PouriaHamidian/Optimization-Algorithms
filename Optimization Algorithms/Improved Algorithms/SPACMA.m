%function [bsf_solution, bsf_fit_var, run_funcvals, runtime] = LSHADE_SPACMA(fhd, problem_size, bounds, pop_size, M)
function [GlobalCost,BestNetWork]=SPACMA(net,Inputs,Targets,run)

%% Problem Definition
problem_size = net.numWeightElements;
CostFunction=@(x) CostANN(x,net,Inputs,Targets);        % Cost Function


tic;
L_Rate= 0.80;
%val_2_reach = 10^(-8);

    pop_size=100;
    max_nfes = 110000;
    %rand('seed', sum(100 * clock));
    lu = [-10 * ones(1, problem_size); 10 * ones(1, problem_size)];
    

        %for run_id = 1 : runs
            nfes = 0;
            run_funcvals = [];
           
            %%  parameter settings for L-SHADE
            p_best_rate = 0.2;    
            arc_rate = 1.4; 
            memory_size = 10; 
%             pop_size = 18 * problem_size;
            max_pop_size = pop_size;
            min_pop_size = 4.0;
            
            %%  parameter settings for Hybridization
            First_calss_percentage=0.5;
            
            %% Initialize the main population
            popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
            pop = popold; % the old population becomes the current population
            
            fitness = [];
            for brojac = 1:size(pop,1)
                fitness(brojac) = CostFunction(pop(brojac,:));
            end
            %fitness = feval(fhd,pop',func);
            fitness = fitness';

            bsf_fit_var = 1e+30;
            bsf_index = 0;
            bsf_solution = zeros(1, problem_size);
            
            %%%%%%%%%%%%%%%%%%%%%%%% for out
            for i = 1 : pop_size
                nfes = nfes + 1;
                if (fitness(i) < bsf_fit_var && isreal(pop(i, :)) && sum(isnan(pop(i, :)))==0 && min(pop(i, :))>=-100 && max(pop(i, :))<=100)
                    bsf_fit_var = fitness(i);
                    bsf_solution = pop(i, :);
                    bsf_index = i;
                end
                
                if nfes > max_nfes;
                    break; 
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%% for out
            
            run_funcvals = [run_funcvals;ones(pop_size,1)*bsf_fit_var];

            
            memory_sf = 0.5 .* ones(memory_size, 1);
            memory_cr = 0.5 .* ones(memory_size, 1);
            memory_pos = 1;
            
            archive.NP = arc_rate * pop_size; % the maximum size of the archive
            archive.pop = zeros(0, problem_size); % the solutions stored in te archive
            archive.funvalues = zeros(0, 1); % the function value of the archived solutions
            
            memory_1st_class_percentage = First_calss_percentage.* ones(memory_size, 1); % Class#1 probability for Hybridization 
            
            %% Initialize CMAES parameters
            sigma = 0.5;          % coordinate wise standard deviation (step size)
            xmean = rand(problem_size,1);    % objective variables initial point
            mu = pop_size/2;               % number of parents/points for recombination
            weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
            mu = floor(mu);
            weights = weights/sum(weights);     % normalize recombination weights array
            mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i
            
            % Strategy parameter setting: Adaptation
            cc = (4 + mueff/problem_size) / (problem_size+4 + 2*mueff/problem_size); % time constant for cumulation for C
            cs = (mueff+2) / (problem_size+mueff+5);  % t-const for cumulation for sigma control
            c1 = 2 / ((problem_size+1.3)^2+mueff);    % learning rate for rank-one update of C
            cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((problem_size+2)^2+mueff));  % and for rank-mu update
            damps = 1 + 2*max(0, sqrt((mueff-1)/(problem_size+1))-1) + cs; % damping for sigma usually close to 1

            % Initialize dynamic (internal) strategy parameters and constants
            pc = zeros(problem_size,1);
            ps = zeros(problem_size,1);   % evolution paths for C and sigma
            B = eye(problem_size,problem_size);                       % B defines the coordinate system
            D = ones(problem_size,1);                      % diagonal D defines the scaling
            C = B * diag(D.^2) * B';            % covariance matrix C
            invsqrtC = B * diag(D.^-1) * B';    % C^-1/2
            eigeneval = 0;                      % track update of B and D
            chiN=problem_size^0.5*(1-1/(4*problem_size)+1/(21*problem_size^2));  % expectation of
            
            
            %% main loop
            Hybridization_flag=1; % Indicator flag if we need to Activate CMAES Hybridization
            
            while nfes < max_nfes
 
                pop = popold; % the old population becomes the current population
                [temp_fit, sorted_index] = sort(fitness, 'ascend');
                
                mem_rand_index = ceil(memory_size * rand(pop_size, 1));
                mu_sf = memory_sf(mem_rand_index);
                mu_cr = memory_cr(mem_rand_index);
                mem_rand_ratio = rand(pop_size, 1);
                
                %% for generating crossover rate
                cr = normrnd(mu_cr, 0.1);
                term_pos = find(mu_cr == -1);
                cr(term_pos) = 0;
                cr = min(cr, 1);
                cr = max(cr, 0);

                %% for generating scaling factor
                if(nfes <= max_nfes/2)
                    sf=0.45+.1*rand(pop_size, 1);
                    pos = find(sf <= 0);
                    
                    while ~ isempty(pos)
                        sf(pos)=0.45+0.1*rand(length(pos), 1);
                        pos = find(sf <= 0);
                    end
                else
                    sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
                    
                    pos = find(sf <= 0);
                    
                    while ~ isempty(pos)
                        sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                        pos = find(sf <= 0);
                    end
                end
                sf = min(sf, 1);
                
                %% for generating Hybridization Class probability
                Class_Select_Index=(memory_1st_class_percentage(mem_rand_index)>=mem_rand_ratio);
                if(Hybridization_flag==0)
                    Class_Select_Index=or(Class_Select_Index,~Class_Select_Index);%All will be in class#1
                end
                
                %%
                r0 = [1 : pop_size];
                popAll = [pop; archive.pop];
                [r1, r2] = gnR1R2(pop_size, size(popAll, 1), r0);
                
                pNP = max(round(p_best_rate * pop_size), 2); %% choose at least two best solutions
                randindex = ceil(rand(1, pop_size) .* pNP); %% select from [1, 2, 3, ..., pNP]
                randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
                pbest = pop(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions
                
                vi=[];
                temp=[];
                if(sum(Class_Select_Index)~=0)
                    vi(Class_Select_Index,:) = pop(Class_Select_Index,:) + sf(Class_Select_Index, ones(1, problem_size)) .* (pbest(Class_Select_Index,:) - pop(Class_Select_Index,:) + pop(r1(Class_Select_Index), :) - popAll(r2(Class_Select_Index), :));
                end
                
                if(sum(~Class_Select_Index)~=0)
                    for k=1:sum(~Class_Select_Index)
                        temp(:,k) = xmean + sigma * B * (D .* randn(problem_size,1)); % m + sig * Normal(0,C)
                    end
                    vi(~Class_Select_Index,:) = temp';
                end
                
                if(~isreal(vi))
                    Hybridization_flag=0;
                    continue;
                end

                
                vi = boundConstraint(vi, pop, lu);

                
                mask = rand(pop_size, problem_size) > cr(:, ones(1, problem_size)); % mask is used to indicate which elements of ui comes from the parent
                rows = (1 : pop_size)'; cols = floor(rand(pop_size, 1) * problem_size)+1; % choose one position where the element of ui doesn't come from the parent
                jrand = sub2ind([pop_size problem_size], rows, cols); mask(jrand) = false;
                ui = vi; ui(mask) = pop(mask);
                
                children_fitness = [];
                for brojac = 1:size(ui,1)
                    children_fitness(brojac) = CostFunction(ui(brojac,:));
                end
%                 children_fitness = feval(fhd, ui', func);
                 children_fitness = children_fitness';
                
                
                %%%%%%%%%%%%%%%%%%%%%%%% for out
                for i = 1 : pop_size
                    nfes = nfes + 1;
                    if (children_fitness(i) < bsf_fit_var && isreal(ui(i, :)) && sum(isnan(ui(i, :)))==0 && min(ui(i, :))>=-100 && max(ui(i, :))<=100)
                        bsf_fit_var = children_fitness(i);
                        bsf_solution = ui(i, :);
                        bsf_index = i;
                    end
                    
                    if nfes > max_nfes;
                        break;
                    end
                end
                %%%%%%%%%%%%%%%%%%%%%%%% for out
                
                run_funcvals = [run_funcvals; ones(pop_size,1)*bsf_fit_var];
                
                dif = abs(fitness - children_fitness);
                
                
                %% I == 1: the parent is better; I == 2: the offspring is better
                Child_is_better_index = (fitness > children_fitness);
                goodCR = cr(Child_is_better_index == 1);
                goodF = sf(Child_is_better_index == 1);
                dif_val = dif(Child_is_better_index == 1);
                dif_val_Class_1 = dif(and(Child_is_better_index,Class_Select_Index) == 1);
                dif_val_Class_2 = dif(and(Child_is_better_index,~Class_Select_Index) == 1);
                
                archive = updateArchive(archive, popold(Child_is_better_index == 1, :), fitness(Child_is_better_index == 1));
                
                [fitness, Child_is_better_index] = min([fitness, children_fitness], [], 2);

                popold = pop;
                popold(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);
                
                num_success_params = numel(goodCR);
                
                if num_success_params > 0
                    sum_dif = sum(dif_val);
                    dif_val = dif_val / sum_dif;
                    
                    %% for updating the memory of scaling factor
                    memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
                    
                    %% for updating the memory of crossover rate
                    if max(goodCR) == 0 || memory_cr(memory_pos)  == -1
                        memory_cr(memory_pos)  = -1;
                    else
                        memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
                    end
                    
                    if (Hybridization_flag==1)% if the Hybridization is activated
                        memory_1st_class_percentage(memory_pos) = memory_1st_class_percentage(memory_pos)*L_Rate+ (1-L_Rate)*(sum(dif_val_Class_1) / (sum(dif_val_Class_1) + sum(dif_val_Class_2)));
                        memory_1st_class_percentage(memory_pos)=min(memory_1st_class_percentage(memory_pos),0.8);
                        memory_1st_class_percentage(memory_pos)=max(memory_1st_class_percentage(memory_pos),0.2);
                    end
                    
                    memory_pos = memory_pos + 1;
                    if memory_pos > memory_size;  memory_pos = 1; end
                end
                
                %% for resizing the population size
                plan_pop_size = round((((min_pop_size - max_pop_size) / max_nfes) * nfes) + max_pop_size);
                
                if pop_size > plan_pop_size
                    reduction_ind_num = pop_size - plan_pop_size;
                    if pop_size - reduction_ind_num <  min_pop_size; reduction_ind_num = pop_size - min_pop_size;end
                    
                    pop_size = pop_size - reduction_ind_num;
                    for r = 1 : reduction_ind_num
                        [valBest, indBest] = sort(fitness, 'ascend');
                        worst_ind = indBest(end);
                        popold(worst_ind,:) = [];
                        pop(worst_ind,:) = [];
                        fitness(worst_ind,:) = [];
                        Child_is_better_index(worst_ind,:) = [];
                    end
                    
                    archive.NP = round(arc_rate * pop_size);
                    
                    if size(archive.pop, 1) > archive.NP
                        rndpos = randperm(size(archive.pop, 1));
                        rndpos = rndpos(1 : archive.NP);
                        archive.pop = archive.pop(rndpos, :);
                    end
                    
                    %% update CMA parameters
                    mu = pop_size/2;               % number of parents/points for recombination
                    weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
                    mu = floor(mu);
                    weights = weights/sum(weights);     % normalize recombination weights array
                    mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i
                end
               
                %% CMAES Adaptation
                if(Hybridization_flag==1)
                    % Sort by fitness and compute weighted mean into xmean
                    [~, popindex] = sort(fitness);  % minimization
                    xold = xmean;
                    xmean = popold(popindex(1:mu),:)' * weights;  % recombination, new mean value
                    
                    % Cumulation: Update evolution paths
                    ps = (1-cs) * ps ...
                        + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma;
                    hsig = sum(ps.^2)/(1-(1-cs)^(2*nfes/pop_size))/problem_size < 2 + 4/(problem_size+1);
                    pc = (1-cc) * pc ...
                        + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;
                    
                    % Adapt covariance matrix C
                    artmp = (1/sigma) * (popold(popindex(1:mu),:)' - repmat(xold,1,mu));  % mu difference vectors
                    C = (1-c1-cmu) * C ...                   % regard old matrix
                        + c1 * (pc * pc' ...                % plus rank one update
                        + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0
                        + cmu * artmp * diag(weights) * artmp'; % plus rank mu update
                    
                    % Adapt step size sigma
                    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));
                    
                    % Update B and D from C
                    if nfes - eigeneval > pop_size/(c1+cmu)/problem_size/10  % to achieve O(problem_size^2)
                        eigeneval = nfes;
                        C = triu(C) + triu(C,1)'; % enforce symmetry
                        if(sum(sum(isnan(C)))>0 || sum(sum(~isfinite(C)))>0 || ~isreal(C))
                            Hybridization_flag=0;
                            continue;
                        end
                        [B,D] = eig(C);           % eigen decomposition, B==normalized eigenvectors
                        D = sqrt(diag(D));        % D contains standard deviations now
                        invsqrtC = B * diag(D.^-1) * B';
                    end
                    
                end
                
            end %%%%%%%%nfes
%run_funcvals = run_funcvals(round(linspace(1,length(run_funcvals),M)));
runtime = toc;

    BestNetWork = ConsNet_Fcn(net,bsf_solution);
GlobalCost=-bsf_fit_var;
fprintf('LSHADE_SPACMA,  run:%d, bestR2:%3.5f \n',run,GlobalCost);        
    



function [r1, r2] = gnR1R2(NP1, NP2, r0)



NP0 = length(r0);

r1 = floor(rand(1, NP0) * NP1) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = (r1 == r0);
    if sum(pos) == 0
        break;
    else % regenerate r1 if it is equal to r0
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    if i > 1000, % this has never happened so far
        error('Can not genrate r1 in 1000 iterations');
    end
end

r2 = floor(rand(1, NP0) * NP2) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = ((r2 == r1) | (r2 == r0));
    if sum(pos)==0
        break;
    else % regenerate r2 if it is equal to r0 or r1
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
    end
    if i > 1000, % this has never happened so far
        error('Can not genrate r2 in 1000 iterations');
    end
end



%This function is used for L-SHADE bound checking 
function vi = boundConstraint (vi, pop, lu)

% if the boundary constraint is violated, set the value to be the middle
% of the previous value and the bound
%

[NP, D] = size(pop);  % the population size and the problem's dimension

%% check the lower bound
xl = repmat(lu(1, :), NP, 1);

pos = vi < xl;
vi(pos) = (pop(pos) + xl(pos)) / 2;

%% check the upper bound
xu = repmat(lu(2, :), NP, 1);
pos = vi > xu;
vi(pos) = (pop(pos) + xu(pos)) / 2;




function archive = updateArchive(archive, pop, funvalue)


if archive.NP == 0, return; end

if size(pop, 1) ~= size(funvalue,1), error('check it'); end

% Method 2: Remove duplicate elements
popAll = [archive.pop; pop ];
funvalues = [archive.funvalues; funvalue ];
[dummy IX]= unique(popAll, 'rows');
if length(IX) < size(popAll, 1) % There exist some duplicate solutions
  popAll = popAll(IX, :);
  funvalues = funvalues(IX, :);
end

if size(popAll, 1) <= archive.NP   % add all new individuals
  archive.pop = popAll;
  archive.funvalues = funvalues;
else                % randomly remove some solutions
  rndpos = randperm(size(popAll, 1)); % equivelent to "randperm";
  rndpos = rndpos(1 : archive.NP);
  
  archive.pop = popAll  (rndpos, :);
  archive.funvalues = funvalues(rndpos, :);
end