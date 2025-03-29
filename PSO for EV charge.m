clc; clear; close all;
rng(0);
%% 参数设定
numEV = 20; T = 24; Pmax = 7; Pmin = -7;
baseLoad = [150, 140, 130, 125, 120, 130, 180, 250, 350, 400, ...
            420, 450, 470, 460, 440, 430, 450, 500, 550, 520, ...
            480, 400, 300, 200]; % 基准负荷波动
lambda = 0.1 + 0.002 * baseLoad;
iterations = 50; pop_size = 30; % 粒子群大小

SOC_max = 50 * ones(numEV, 1); SOC_min = 10 * ones(numEV, 1);
SOC_init = randi([15, 40], numEV, 1);
SOC_target = randi([30, 45], numEV, 1);
EV_arrival = randi([1 12], numEV, 1);
EV_departure = EV_arrival + randi([6 12], numEV, 1);

EV_status = zeros(numEV, T); % 初始化矩阵

for i = 1:numEV
    EV_status(i, EV_arrival(i):min(EV_departure(i), T)) = 1;
end

w1 = 1; w2 = 0.5; w3 = 5; % 目标权重

%% 粒子群优化参数
w = 0.7; c1 = 1.5; c2 = 1.5;
v_max = (Pmax - Pmin) / 2;

EV_status_expanded = repmat(EV_status, [1, 1, pop_size]);
EV_status_expanded = permute(EV_status_expanded, [3, 1, 2]); % 变换维度，使其变为 pop_size × 20 × 24

% 初始化粒子群
pop = zeros(pop_size, numEV, T);  % 充放电功率矩阵
velocity = zeros(pop_size, numEV, T);  % 速度矩阵
fitness_values = inf(pop_size, 1); % 记录适应度值

% 逐个 EV 进行初始化，确保 SOC 约束
for ev = 1:numEV
    for p = 1:pop_size
        % 可充放电时间段
        active_slots = find(EV_status(ev, :) == 1);
        num_slots = length(active_slots);
        
        % 计算需要充的电量
        energy_needed = SOC_target(ev) - SOC_init(ev);
        
        % 计算每个时间步的功率，确保 SOC 不超限
        if energy_needed > 0
            P_avg = energy_needed / num_slots; 
            P_values = P_avg + 0.2 * (rand(1, num_slots) - 0.5) * P_avg; % 限制扰动
        else
            P_values = zeros(1, num_slots); % 目标已满足，无需充电
        end

        % 限制在充放电范围 [Pmin, Pmax]
        P_values = max(min(P_values, Pmax), Pmin);

        % 计算最终 SOC，确保不超限
        estimated_SOC = SOC_init(ev) + sum(P_values);
        if estimated_SOC > SOC_max(ev)
            P_values = P_values * (SOC_max(ev) - SOC_init(ev)) / (estimated_SOC - SOC_init(ev));
        elseif estimated_SOC < SOC_min(ev)
            P_values = P_values * (SOC_min(ev) - SOC_init(ev)) / (estimated_SOC - SOC_init(ev));
        end

        % 赋值到种群
        pop(p, ev, active_slots) = P_values;
        
        % 速度初始化（随机扰动）
        velocity(p, ev, active_slots) = (rand(1, num_slots) * (2 * v_max) - v_max);
    end
end

% 计算初始适应度值
for p = 1:pop_size
    P_ev = squeeze(pop(p, :, :)); 
    totalLoad = sum(P_ev, 1) + baseLoad;

    % 目标函数计算
    J1 = sum((totalLoad - mean(baseLoad)).^2);
    J2 = sum(sum(lambda .* P_ev));
    SOC_final = SOC_init + sum(P_ev, 2);
    J3 = sum((SOC_target - SOC_final).^2);

    fitness_values(p) = w1 * J1 + w2 * J2 + w3 * J3;
end

% 设定最优个体
[~, best_idx] = min(fitness_values);
g_best = pop(best_idx, :, :);
p_best = pop;
p_best_val = fitness_values;
g_best_val = fitness_values(best_idx);
J_values = zeros(iterations, 1);

%% 迭代优化
for iter = 1:iterations
    fitness = zeros(pop_size, 1);
    
    for i = 1:pop_size
        P_ev = squeeze(pop(i, :, :));
        totalLoad = sum(P_ev, 1) + baseLoad;

        % 目标函数计算
        J1 = sum((totalLoad - mean(baseLoad)).^2);
        J2 = sum(sum(lambda .* P_ev));
        SOC_final = SOC_init + sum(P_ev, 2);
        penalty = sum(max(0, SOC_target - SOC_final));
        J3 = sum((SOC_target - SOC_final).^2);%这样会导致有的EV充电能量不满足要求

        fitness(i) = w1 * J1 + w2 * J2 + w3 * J3 + 10 * penalty;

        % 更新个体最优值
        if fitness(i) < p_best_val(i)   %  与该种群的历史最优适应度值比较
            p_best(i, :, :) = pop(i, :, :);     %  该种群的位置更新
            p_best_val(i) = fitness(i);    % 该种群的最优适应度值更新
        end
    end
    
    % 更新全局最优值
    [min_val, min_idx] = min(fitness);
    if min_val < g_best_val
        g_best = pop(min_idx, :, :);
        g_best_val = min_val;
    end
    
    
    % 更新粒子速度和位置
    for i = 1:pop_size
        velocity(i, :, :) = w * velocity(i, :, :) ...
                        + c1 * rand() * (p_best(i, :, :) - pop(i, :, :)) ...
                        + c2 * rand() * (g_best - pop(i, :, :));
        
        % 限制速度范围
        velocity(i, :, :) = max(min(velocity(i, :, :), v_max), -v_max);
        
        % 更新位置
        pop(i, :, :) = pop(i, :, :) + velocity(i, :, :);
        
        % 限制充电功率范围
        pop(i, :, :) = max(min(pop(i, :, :), Pmax), Pmin);

        % 计算新的SOC
        P_ev = squeeze(pop(i, :, :));
        SOC_new = SOC_init + cumsum(P_ev, 2);%得到所有EV各时刻的电池能量信息
    
        % **确保SOC不会超出上下界**
        SOC_new = max(min(SOC_new, SOC_max), SOC_min);

        % **修正超出SOC范围的P_ev**
        for ev = 1:numEV
            for t = 1:T
                % 计算当前时刻的SOC
                if t == 1
                    SOC_current = SOC_init(ev) + P_ev(ev, t);
                else
                    SOC_current = SOC_new(ev, t-1) + P_ev(ev, t);
                end
        
                % 计算该时刻的可充放电范围
                P_max_allowed = SOC_max(ev) - SOC_current;
                P_min_allowed = SOC_min(ev) - SOC_current;

                % 约束P_ev在合法范围内
                P_ev(ev, t) = max(min(P_ev(ev, t), P_max_allowed), P_min_allowed);
        
                % 更新SOC
                if t == 1
                    SOC_new(ev, t) = SOC_init(ev) + P_ev(ev, t);
                else
                    SOC_new(ev, t) = SOC_new(ev, t-1) + P_ev(ev, t);
                end
            end
        end

    
        % **确保EV离开时SOC至少满足SOC_target**
        for ev = 1:numEV
            if SOC_new(ev, EV_departure(ev)) < SOC_target(ev)
                % 计算缺少的SOC
                deficit = SOC_target(ev) - SOC_new(ev, EV_departure(ev));
                % 找到EV停车期间的时间步
                active_times = find(EV_status(ev, :) == 1);  %   indices
                num_active = length(active_times);
                if num_active > 0
                    % 在停车时段均匀增加充电量，同时确保不会超过SOC_max
                    for t = active_times
                        if deficit <= 0
                            break; % 充足后退出
                        end
                        % 计算该时刻还能充多少
                        max_charge = SOC_max(ev) - SOC_new(ev, t);
                        charge_addition = min(deficit / num_active, max_charge);
                        P_ev(ev, t) = P_ev(ev, t) + charge_addition;
                        % 重新计算SOC
                        SOC_new(ev, t:end) = SOC_new(ev, t:end) + charge_addition;
                        % 更新剩余的充电需求
                        deficit = deficit - charge_addition;
                    end
                end
            end
        end
    end
        % **更新pop**
        pop(i, :, :) = P_ev;
    
    
    
    % 记录目标函数
    J_values(iter) = g_best_val;
end
%% SOC

g_best=squeeze(g_best);
SOC_new=SOC_init + cumsum(g_best, 2);
% SOC_matrix=zeros(numEV, T+1);
% SOC_matrix(:,1)=SOC_init;
% for ii=2:T+1
%     for jj=1:numEV
%         SOC_matrix(jj,ii)=SOC_matrix(jj,ii-1)+g_best(jj,ii-1);
%     end
% end

%% 绘图
figure;
subplot(5,1,1);
plot(1:T, sum(squeeze(g_best), 1) + baseLoad, 'b', 'LineWidth', 2); hold on;
plot(1:T, baseLoad, 'r--', 'LineWidth', 2);
xlabel('时间 (小时)'); ylabel('负荷 (kW)');
title('优化后总负荷'); legend('优化后', '基准负荷');

subplot(5,1,2);
plot(1:iterations, J_values, 'm', 'LineWidth', 2);
xlabel('迭代次数'); ylabel('目标函数');
title('优化目标收敛曲线');

subplot(5,1,3);
imagesc(squeeze(g_best)); colormap(jet); colorbar;
xlabel('时间 (小时)'); ylabel('EV编号');
title('EV 充放电策略');

subplot(5,1,4);
plot(1:numEV, SOC_final, 'k', 'LineWidth', 2);
xlabel('EV no.'); ylabel('电池能量');
title('EV充电结束能量');

subplot(5,1,5);
plot(1:T, SOC_new', 'k', 'LineWidth', 2);
xlabel('时间（小时）'); ylabel('电池能量');
title('EV能量变化');