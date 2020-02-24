//Contributors: Sibo Wang, Renchi Yang
#ifndef FORA_QUERY_H
#define FORA_QUERY_H

#include "algo.h"
#include "graph.h"
#include "heap.h"
#include "config.h"
#include "build.h"
#include "string"
//#define CHECK_PPR_VALUES 1
//#define PRINT_PPR_VALUES 1
// #define CHECK_TOP_K_PPR 1
#define PRINT_PRECISION_FOR_DIF_K 1
// std::mutex mtx;

void montecarlo_query_dht(int v, const Graph &graph) {
    Timer timer(MC_DHT_QUERY);

    rw_counter.reset_zero_values();
    dht.reset_zero_values();
    unordered_map<int, pair<int, int>> occur;

    {
        Timer tm(RONDOM_WALK);
        INFO(config.omega);
        num_total_rw += config.omega;
        for (unsigned long i = 0; i < config.omega; i++) {
            random_walk_dht(v, i, graph, occur);
        }
    }

    for (auto item:occur) {
        int node_id = item.first;
        dht[node_id] = item.second.second * 1.0 / config.omega;
    }

#ifdef CHECK_PPR_VALUES
    display_dht();
#endif
}

void montecarlo_query_dht_topk(int v, const Graph &graph) {
    Timer timer(0);

    rw_counter.reset_zero_values();
    unordered_map<int, pair<int, int>> occur;
    // random walk phase
    {
        Timer tm(RONDOM_WALK);
        INFO(config.omega);
        num_total_rw += config.omega;
        for (unsigned long i = 0; i < config.omega; i++) {
            random_walk_dht(v, i, graph, occur);
        }
    }
    // compute dht by random walk paths
    dht.clean();
    for (auto item:occur) {
        int node_id = item.first;
        dht.insert(node_id, item.second.second * 1.0 / config.omega);
    }

#ifdef CHECK_PPR_VALUES
    display_dht();
#endif
}

void global_iteration_query(int v, const Graph &graph) {
    Timer timer(GI_QUERY);

    dht.reset_zero_values();
    vector<double> dht_tmp(graph.n, 0);
    for (int l = 0; l < dht.m_num; ++l) {
        double max_error = 0;
        bool first = true;
        int k = 0;
        dht_tmp.clear();
        //vector<int> dht_to_l;
        do {
            k++;
            global_iteration(v, dht_tmp, graph);
            if (first) {
                first = false;
                max_error = *max_element(dht_tmp.begin(), dht_tmp.end());
            }
        } while (pow(1 - config.alpha, k) / config.alpha * max_error > config.epsilon * config.delta);
        dht.insert(l, dht_tmp[l]);
    }
#ifdef CHECK_PPR_VALUES
    display_dht();
#endif
}

void dne_query(int v, const Graph &graph) {
    Timer timer(DNE_QUERY);

    dht.reset_zero_values();
    int dne_m = 30000;
    INFO(dne_m);
    dhe_query_basic(v, v, dne_m, graph);
#ifdef CHECK_PPR_VALUES
    display_dht();
#endif
}
//compute ppr(t,t) if ppr(s,t) larger than threshold and get dht
void bippr_query_with_threshold(const Graph &graph, double threshold) {
    ppr_bi.initialize(graph.n);
    //ppr_self
    rw_counter.initialize(graph.n);
    fill(bwd_idx.first.occur.m_data, bwd_idx.first.occur.m_data + graph.n, -1);
    fill(bwd_idx.second.occur.m_data, bwd_idx.second.occur.m_data + graph.n, -1);
    fill(rw_counter.occur.m_data, rw_counter.occur.m_data + graph.n, -1);
    vector<int> idx(graph.n, -1), node_with_r(graph.n), q(graph.n);
    int pointer_r = 0;
    int total_rw_b = 0;
    //static unordered_map<int, int > idx;///
    INFO(config2.omega);
    assert(config2.rmax < 1);
    unsigned long long backward_counter = 0, old_num_total_rw = num_total_rw;
    int counter = 0;
    for (int k = 0; k < ppr.m_num; ++k) {
        if (ppr[k] <= threshold)continue;
        counter++;
        {
            Timer tm(RONDOM_WALK);
            assert(residual < config.rmax);
            assert(residual < config2.rmax);
            unsigned long num_s_rw = ceil(config2.omega);
            num_total_rw += num_s_rw;
            total_rw_b += num_s_rw;
            for (unsigned long i = 0; i < num_s_rw; i++) {
                int destination = random_walk(k, graph);
                if (rw_counter.occur[destination] != k) {
                    rw_counter.occur[destination] = k;
                    rw_counter[destination] = 1;
                } else {
                    ++rw_counter[destination];
                }
            }
        }
        if (config2.rmax < 1.0) {
            Timer tm(BWD_LU);
            backward_counter += reverse_local_update_linear_dht(k, graph, idx, node_with_r, pointer_r, q);
            assert(bwd_idx.first[k] > 0 && bwd_idx.first.occur[k] == k);
            ppr_bi.insert(k, bwd_idx.first[k]);
            for (int j = 0; j < pointer_r; ++j) {
                int nodeid = node_with_r[j];
                double residual = bwd_idx.second[nodeid];
                if (rw_counter.occur[nodeid] == k) {
                    ppr_bi[k] += rw_counter[nodeid] * 1.0 / config2.omega * residual;
                }
            }
        } else {
            assert(rw_counter.occur[k] == k);
            if (rw_counter.occur[k] == k) {
                ppr_bi.insert(k, rw_counter[k] / 1.0 / config2.omega);
            }
        }
    }
    dht.initialize(graph.n);
    for (int l = 0; l < ppr_bi.occur.m_num; ++l) {
        int nodeid = ppr_bi.occur[l];
        dht.insert(nodeid, ppr[nodeid] / ppr_bi[nodeid]);
    }
    INFO(counter);
    INFO(total_rw_b);
}

void compute_ppr_with_reserve() {
    ppr.clean();
    int node_id;
    double reserve;
    for (long i = 0; i < fwd_idx.first.occur.m_num; i++) {
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[node_id];
        if (reserve)
            ppr.insert(node_id, reserve);
    }
}

void bippr_query_with_fora(const Graph &graph, double check_rsum) {
    ppr.reset_zero_values();
    int node_id;
    double reserve;
    for (long i = 0; i < fwd_idx.first.occur.m_num; i++) {
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[node_id];
        ppr[node_id] = reserve;
    }
    if (check_rsum == 0.0)
        return;

    unsigned long long num_random_walk = config.omega * check_rsum;
    INFO(num_random_walk);

    //ppr_self
    ppr_bi.initialize(graph.n);
    rw_counter.initialize(graph.n);
    fill(bwd_idx.first.occur.m_data, bwd_idx.first.occur.m_data + graph.n, -1);
    fill(bwd_idx.second.occur.m_data, bwd_idx.second.occur.m_data + graph.n, -1);
    fill(rw_counter.occur.m_data, rw_counter.occur.m_data + graph.n, -1);
    vector<int> idx(graph.n, -1), node_with_r(graph.n), q(graph.n);
    int pointer_r = 0;
    INFO(config2.omega);
    assert(config2.rmax < 1);
    unsigned long long backward_counter = 0, old_num_total_rw = num_total_rw;
    int counter = 0;
    unsigned long num_s_rw_real;
    for (int k = 0; k < ppr.m_num; ++k) {
        if (ppr[k] <= config.delta && (fwd_idx.second.notexist(k) || fwd_idx.second[k] <= 0))continue;
        counter++;
        {
            Timer tm(RONDOM_WALK);
            double residual = fwd_idx.second[k] > 0 ? fwd_idx.second[k] : 0;
            unsigned long num_s_rw = ceil(residual / check_rsum * num_random_walk);
            if (ppr[k]>config.delta)
                num_s_rw_real = ceil(config2.omega) > num_s_rw ? ceil(config2.omega) : num_s_rw;
            else
                num_s_rw_real=num_s_rw;

            double ppr_incre = residual / num_s_rw_real;
            num_total_rw += num_s_rw_real;
            for (unsigned long i = 0; i < num_s_rw_real; i++) {
                int destination = random_walk(k, graph);
                ppr[destination] += ppr_incre;
                if (ppr[k]>config.delta){
                    if (rw_counter.occur[destination] != k) {
                        rw_counter.occur[destination] = k;
                        rw_counter[destination] = 1;
                    } else {
                        ++rw_counter[destination];
                    }
                }
            }
        }
        if (ppr[k]<=config.delta) continue;
        if (config2.rmax < 1.0) {
            Timer tm(BWD_LU);
            backward_counter += reverse_local_update_linear_dht(k, graph, idx, node_with_r, pointer_r, q);
            assert(bwd_idx.first[k] > 0 && bwd_idx.first.occur[k] == k);
            ppr_bi.insert(k, bwd_idx.first[k]);
            for (int j = 0; j < pointer_r; ++j) {
                int nodeid = node_with_r[j];
                double residual = bwd_idx.second[nodeid];
                if (rw_counter.occur[nodeid] == k) {
                    ppr_bi[k] += rw_counter[nodeid] * 1.0 / num_s_rw_real * residual;
                }
            }
        } else {
            assert(rw_counter.occur[k] == k);
            if (rw_counter.occur[k] == k) {
                ppr_bi.insert(k, rw_counter[k] / 1.0 / num_s_rw_real);
            }
        }
    }
    for (int m = 0; m < ppr.m_num; ++m) {
        if (ppr[m]>config.delta&&ppr_bi.notexist(m)){
            {
                Timer tm(RONDOM_WALK);
                num_s_rw_real= ceil(config2.omega);
                num_total_rw += num_s_rw_real;
                for (unsigned long i = 0; i < num_s_rw_real; i++) {
                    int destination = random_walk(m, graph);
                    if (rw_counter.occur[destination] != m) {
                        rw_counter.occur[destination] = m;
                        rw_counter[destination] = 1;
                    } else {
                        ++rw_counter[destination];
                    }
                }
            }
            if (config2.rmax < 1.0) {
                Timer tm(BWD_LU);
                backward_counter += reverse_local_update_linear_dht(m, graph, idx, node_with_r, pointer_r, q);
                ppr_bi.insert(m, bwd_idx.first[m]);
                for (int j = 0; j < pointer_r; ++j) {
                    int nodeid = node_with_r[j];
                    double residual = bwd_idx.second[nodeid];
                    if (rw_counter.occur[nodeid] == m) {
                        ppr_bi[m] += rw_counter[nodeid] * 1.0 / num_s_rw_real * residual;
                    }
                }
            } else {
                if (rw_counter.occur[m] == m) {
                    ppr_bi.insert(m, rw_counter[m] / 1.0 / num_s_rw_real);
                }
            }
        }
    }
    dht.initialize(graph.n);
    for (int l = 0; l < ppr_bi.occur.m_num; ++l) {
        int nodeid = ppr_bi.occur[l];
        dht.insert(nodeid, ppr[nodeid] / ppr_bi[nodeid]);
    }
    num_total_bi += backward_counter;
}

int bippr_query_candidate_with_idx(const Graph &graph, const unordered_map<int, bool> &candidate, const double &lowest_rmax,
                          unordered_map<int, vector<int>> &backward_from) {
    Timer timer(BIPPR_QUERY);
    ppr_bi.clean();
    static vector<int> in_backward(graph.n);
    static vector<int> in_next_backward(graph.n);
    std::fill(in_backward.begin(), in_backward.end(), -1);
    std::fill(in_next_backward.begin(), in_next_backward.end(), -1);
    fill(rw_counter.occur.m_data, rw_counter.occur.m_data + graph.n, -1);
    unsigned long long backward_counter = 0, old_num_total_rw = num_total_rw;
    for (auto item:candidate) {
        int node_id = item.first;
        if (config2.rmax < 1.0) {
            Timer tm(BWD_LU);
            backward_counter += reverse_local_update_linear_dht_topk(node_id, graph, lowest_rmax, in_backward,
                                                                     in_next_backward, backward_from);
            ppr_bi.insert(node_id, multi_bwd_idx_p[node_id]);
            {
                Timer tm(DFS_CYCLE);
                int num_s_rw=ceil(config2.omega);
                num_total_rw += num_s_rw;
                int index_size=rw_index[node_id].size();
                if (num_s_rw>index_size){
                    for (int j = 0; j < index_size; ++j) {
                        int des =rw_index[node_id][j];
                        if (multi_bwd_idx_r[node_id].find(des)!=multi_bwd_idx_r[node_id].end()){
                            ppr_bi[node_id] += multi_bwd_idx_r[node_id][des]/num_s_rw;
                        }
                    }
                    for (int k = 0; k < num_s_rw - index_size; ++k) {
                        int des = random_walk(node_id, graph);
                        rw_index[node_id].emplace_back(des);
                        if (multi_bwd_idx_r[node_id].find(des)!=multi_bwd_idx_r[node_id].end()){
                            ppr_bi[node_id] += multi_bwd_idx_r[node_id][des]/num_s_rw;
                        }
                    }
                }else{
                    for (int j = 0; j < num_s_rw; ++j) {
                        int des = rw_index[node_id][j];
                        if (multi_bwd_idx_r[node_id].find(des)!=multi_bwd_idx_r[node_id].end()){
                            ppr_bi[node_id] += multi_bwd_idx_r[node_id][des]/num_s_rw;
                        }
                    }
                }
            }
        } else {
            {
                Timer tm(DFS_CYCLE);
                int num_s_rw=ceil(config2.omega);
                num_total_rw += num_s_rw;
                int index_size=rw_index[node_id].size();
                ppr_bi.insert(node_id, 0);
                //INFO(node_id,ppr_bi[node_id]);
                if (num_s_rw>index_size){
                    for (int j = 0; j < index_size; ++j) {
                        int des =rw_index[node_id][j];
                        if (des==node_id){
                            ppr_bi[node_id] += 1.0/num_s_rw;
                        }
                    }
                    for (int k = 0; k < num_s_rw - index_size; ++k) {
                        int des = random_walk(node_id, graph);
                        rw_index[node_id].emplace_back(des);
                        if (des==node_id){
                            ppr_bi[node_id] += 1.0/num_s_rw;
                        }
                    }
                }else{
                    for (int j = 0; j < num_s_rw; ++j) {
                        int des = rw_index[node_id][j];
                        if (des==node_id){
                            ppr_bi[node_id] += 1.0/num_s_rw;
                        }
                    }
                }
                //INFO(node_id,ppr_bi[node_id]);
            }
        }
    }
    set_ppr_self_bounds(graph, candidate);
    num_total_bi += backward_counter;
    return backward_counter;
}


void compute_ppr_with_fwdidx(const Graph &graph, double check_rsum) {
    ppr.reset_zero_values();
    int node_id;
    double reserve;
    for (long i = 0; i < fwd_idx.first.occur.m_num; i++) {
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[node_id];
        ppr[node_id] = reserve;
    }

    if (check_rsum == 0.0)
        return;
    INFO(check_rsum);
    unsigned long long num_random_walk = config.omega * check_rsum;
    INFO(num_random_walk);
    int real_random_walk_f = 0;

    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
        for (long i = 0; i < fwd_idx.second.occur.m_num; i++) {
            int source = fwd_idx.second.occur[i];
            double residual = fwd_idx.second[source];
            unsigned long num_s_rw = ceil(residual / check_rsum * num_random_walk);
            double a_s = residual / check_rsum * num_random_walk / num_s_rw;

            double ppr_incre = a_s * check_rsum / num_random_walk;

            num_total_rw += num_s_rw;
            real_random_walk_f += num_s_rw;
            for (unsigned long j = 0; j < num_s_rw; j++) {
                int des = random_walk(source, graph);
                ppr[des] += ppr_incre;
            }
        }
    }
    INFO(real_random_walk_f);
}


int compute_ppr_with_fwdidx_topk_with_bound_with_idx(const Graph &graph, double check_rsum,
                                            unordered_map<int, vector<int>> &rw_saver) {
    compute_ppr_with_reserve();

    if (check_rsum == 0.0)
        return 0;

    long num_random_walk = config.omega * check_rsum;
    long real_num_rand_walk = 0;
    int count_rw_fora_iter = 0, map_counter = 0;
    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk

        //Timer tm(SOURCE_DIST);
        { //rand walk online
            for (long i = 0; i < fwd_idx.second.occur.m_num; i++) {
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                long num_s_rw = ceil(residual / check_rsum * num_random_walk);
                double a_s = residual / check_rsum * num_random_walk / num_s_rw;

                real_num_rand_walk += num_s_rw;
                num_total_rw += num_s_rw;
                double ppr_incre = a_s * check_rsum / num_random_walk;
                int index_size=rw_index[source].size();
                // reuse random walk paths
                if (num_s_rw>index_size){
                    for (int j = 0; j < index_size; ++j) {
                        int des =rw_index[source][j];
                        if (!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                    for (int k = 0; k < num_s_rw - index_size; ++k) {
                        int des = random_walk(source, graph);
                        rw_index[source].emplace_back(des);
                        if (!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                }else{
                    for (int j = 0; j < num_s_rw; ++j) {
                        int des = rw_index[source][j];
                        if (!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                }
            }
        }
    }
    if (config.delta < threshold)
        set_ppr_bounds(graph, check_rsum, real_num_rand_walk);
    return real_num_rand_walk;
}
void fbrw_query(int v, const Graph &graph, double raw_epsilon) {
    Timer timer(FB_QUERY);
    double rsum = 1.0, ratio = graph.n;//sqrt(config.alpha);
    fora_bippr_setting(graph.n, graph.m, ratio, raw_epsilon);
    static vector<int> forward_from;
    forward_from.reserve(graph.n);
    forward_from.push_back(v);
    //fora_bippr_setting(graph.n, graph.m, 1,raw_epsilon);
    {
        Timer timer(FORA_QUERY);
        {
            Timer timer(FWD_LU);
            fwd_idx.first.clean();  //reserve
            fwd_idx.second.clean();  //residual
            fwd_idx.second.insert(v, rsum);
            long forward_push_num = 0;
            bool flag= true,first= true;
            do {
                //cout << endl;
                //display_setting();
                if (!first)
                    flag=check_cost(rsum, ratio, graph.n, graph.m, forward_push_num, raw_epsilon);
                first= false;
                forward_push_num += forward_local_update_linear_topk_dht2(v, graph, rsum, config.rmax, 0, forward_from);
            } while (flag);
        }
        bippr_query_with_fora(graph, rsum);
    }
}

void fbrw_topk(int v, const Graph &graph) {
    //ppr fwd_idx.first fwd_idx.second的occur都是有效的
    Timer timer(0);
    const static double min_delta = config.alpha / graph.n;
    INFO(min_delta);
    const static double init_delta = 1.0 / 4;
    const static double min_epsilon = sqrt(1.0 / graph.n / config.alpha);
    const static double raw_epsilon = config.epsilon;
    threshold = (1.0 - config.ppr_decay_alpha) / pow(500, config.ppr_decay_alpha) /
                pow(graph.n, 1 - config.ppr_decay_alpha);
    const static double new_pfail = 1.0 / graph.n / graph.n / log(graph.n);

    config.pfail = new_pfail;  // log(1/pfail) -> log(1*n/pfail)
    config.pfail /= 2;
    config2 = config;
    //config.epsilon = config.epsilon / (2 + config.epsilon);//(1+new_epsilon)/(1-new_epsilon)<=1+epsilon
    double ratio_f_b = sqrt(config.alpha * graph.n / threshold) / config.k;
    config2.delta = config.alpha;
    config.delta = init_delta;

    const static double lowest_delta_rmax =
            config.epsilon * sqrt(min_delta / 3 / graph.n / log(2 / new_pfail));//更新后的r_max
    const static double lowest_delta_rmax_2 =
            min_epsilon * sqrt(config2.delta / 3 / log(2.0 / new_pfail));

    double rsum = 1.0;
    static vector<int> forward_from;
    forward_from.clear();
    forward_from.reserve(graph.n);
    forward_from.push_back(v);
    unordered_map<int, vector<int>> backward_from;

    fwd_idx.first.clean();  //reserve
    fwd_idx.second.clean();  //residual
    fwd_idx.second.insert(v, rsum);
    multi_bwd_idx_p.clear();
    multi_bwd_idx_r.clear();
    //init bounds for ppr(s,t), ppr(t,t) dht(s,t)
    upper_bounds.reset_one_values();
    lower_bounds.reset_zero_values();
    init_bounds_self(graph);
    upper_bounds_dht.reset_one_values();
    lower_bounds_dht.reset_zero_values();
    //set<int> candidate;
    unordered_map<int, bool> candidate;
    candidate.clear();
    display_setting();
    bool run_fora = true;
    unordered_map<int, vector<int>> rw_saver;

    num_total_fo = 0;
    num_total_bi = 0;
    int init_rw_num = num_total_rw;
    while (config.delta >= min_delta) {
        double old_f_rmax = config.rmax, old_b_rmax = config2.rmax, forward_counter, backward_counter, real_num_rand_walk;
        int old_candidate_size = candidate.size();
        fora_bippr_setting(graph.n, graph.m, 1, 1, true);
        if (run_fora) {
            num_iter_topk++;
            {
                Timer timer(FWD_LU);
                forward_counter = forward_local_update_linear_topk_dht2(v, graph, rsum, config.rmax, lowest_delta_rmax,
                                                                        forward_from); //forward propagation, obtain reserve and residual
            }
            real_num_rand_walk = compute_ppr_with_fwdidx_topk_with_bound_with_idx(graph, rsum, rw_saver);
        } else {
            backward_counter = bippr_query_candidate_with_idx(graph, candidate, lowest_delta_rmax_2, backward_from);
        }
        if (config.delta < threshold) {
            set_dht_bounds(candidate);
        }
        //INFO(config.delta, config.rmax, config.omega, fwd_idx.second.occur.m_size, candidate.size());
        //compute_ppr_with_fwdidx_topk_with_bound(graph, rsum);
        if (if_stop_dht(candidate, raw_epsilon, min_delta) ||
            (config.delta <= min_delta && config2.epsilon <= min_epsilon)) {

            dht.clean();
            for (auto item:candidate) {
                int node = item.first;
                dht.insert(node, ppr[node] / ppr_bi[node]);
            }
            break;
        } else if (config.delta > min_delta && config2.epsilon > min_epsilon) {
            run_fora = test_run_fora(candidate.size(), graph.m, graph.n, forward_counter, real_num_rand_walk);
            if (run_fora) {
                config.delta = max(min_delta, config.delta / 2.0);  // otherwise, reduce delta to delta/2
            } else {
                config2.epsilon /= 2;
            }
        } else if (config.delta > min_delta) {
            run_fora = test_run_fora(candidate.size(), graph.m, graph.n, forward_counter, real_num_rand_walk);
            if (run_fora == false) {
                dht.clean();
                for (auto item:candidate) {
                    int node = item.first;
                    dht.insert(node, ppr[node] / ppr_bi[node]);
                }
                break;
            }
            config.delta = max(min_delta, config.delta / 2.0);  // otherwise, reduce delta to delta/2
        } else {
            if (config2.epsilon == raw_epsilon) {
                config2.epsilon = max(min_epsilon, config2.epsilon / 2);
                run_fora = false;
            } else {
                if (old_candidate_size > candidate.size() * 1.01) {
                    config2.epsilon = max(min_epsilon, config2.epsilon / 2);
                    run_fora = false;
                    continue;
                }
                run_fora = test_run_fora(candidate.size(), graph.m, graph.n, forward_counter, real_num_rand_walk);
                if (run_fora == true) {
                    dht.clean();
                    for (auto item:candidate) {
                        int node = item.first;
                        dht.insert(node, ppr[node] / ppr_bi[node]);
                    }
                    break;
                } else {
                    config2.epsilon = max(min_epsilon, config2.epsilon / 2);
                }
            }
        }
    }
    INFO(num_total_rw - init_rw_num);
    return;

}


void fora_bippr_query(int v, const Graph &graph, bool topk = false) {
    Timer tm(FBRAW_QUERY);
    double rsum = 1.0, threshold;
    display_setting();
    {
        //forward propagation, obtain reserve and residual
        Timer timer(FWD_LU);
        forward_local_update_linear(v, graph, rsum, config.rmax);
    }
    compute_ppr_with_fwdidx(graph, rsum);
    if (topk) {
        // each node larger than kth ppr(s,v)*alpha is candidate
        static vector<double> temp_ppr;
        temp_ppr.clear();
        temp_ppr.resize(graph.n);
        int size = 0;
        for (int i = 0; i < graph.n; i++) {
            if (ppr.m_data[i] > 0)
                temp_ppr[size++] = ppr.m_data[i];
        }
        nth_element(temp_ppr.begin(), temp_ppr.begin() + config.k - 1, temp_ppr.end(), cmp);
        threshold = temp_ppr[config.k - 1] * config.alpha;
    } else {
        threshold = config.delta;
    }
    bippr_query_with_threshold(graph, threshold);
}



void get_topk(int v, Graph &graph) {

    display_setting();
    if (config.algo == MC) {
        montecarlo_query_dht_topk(v, graph);
        topk_dht();
    } else if (config.algo == DNE) {
        Timer timer(0);
        dne_query(v, graph);
    } else if (config.algo == FORA_BIPPR) {
        fora_bippr_query(v, graph, true);
        topk_dht();
    } else if (config.algo == FBRW) {
        rw_index.clear();
        fbrw_topk(v, graph);
        topk_dht();
        topk_ppr();
    }

    /// check top-k results for different k
    if (config.algo == MC) {
        compute_precision_for_dif_k_dht(v);
    }

    compute_precision_dht(v);

}
void fwd_power_iteration(const Graph &graph, int start, unordered_map<int, double> &map_ppr) {
    static thread_local unordered_map<int, double> map_residual;
    map_residual[start] = 1.0;

    int num_iter = 0;
    double rsum = 1.0;
    while (num_iter < config.max_iter_num) {
        num_iter++;
        // INFO(num_iter, rsum);
        vector<pair<int, double> > pairs(map_residual.begin(), map_residual.end());
        map_residual.clear();
        for (const auto &p: pairs) {
            if (p.second > 0) {
                map_ppr[p.first] += config.alpha * p.second;
                int out_deg = graph.g[p.first].size();

                double remain_residual = (1 - config.alpha) * p.second;
                rsum -= config.alpha * p.second;
                if (out_deg == 0) {
                    map_residual[start] += remain_residual;
                } else {
                    double avg_push_residual = remain_residual / out_deg;
                    for (int next : graph.g[p.first]) {
                        map_residual[next] += avg_push_residual;
                    }
                }
            }
        }
        pairs.clear();
    }
    map_residual.clear();
}

void fwd_power_iteration_self(const Graph &graph, int start, Bwdidx &bwd_idx_th, vector<int> &idx, vector<int> &q) {
    //static thread_local unordered_map<int, double> map_residual;
    int pointer_q = 0;
    int left = 0;
    double myeps = 1.0 / 10000000000;
    q[pointer_q++] = start;
    bwd_idx_th.second.occur[start] = start;
    bwd_idx_th.second[start] = 1;

    idx[start] = start;
    while (left != pointer_q) {
        int v = q[left++];
        left %= graph.n;
        idx[v] = -1;
        if (bwd_idx_th.second[v] < myeps)
            break;

        if (bwd_idx_th.first.occur[v] != start) {
            bwd_idx_th.first.occur[v] = start;
            bwd_idx_th.first[v] = bwd_idx_th.second[v] * config2.alpha;
        } else
            bwd_idx_th.first[v] += bwd_idx_th.second[v] * config.alpha;

        double residual = (1 - config2.alpha) * bwd_idx_th.second[v];
        bwd_idx_th.second[v] = 0;
        if (graph.gr[v].size() > 0) {
            for (int next : graph.gr[v]) {
                int cnt = graph.g[next].size();
                if (bwd_idx_th.second.occur[next] != start) {
                    bwd_idx_th.second.occur[next] = start;
                    bwd_idx_th.second[next] = residual / cnt;
                } else
                    bwd_idx_th.second[next] += residual / cnt;

                if (bwd_idx_th.second[next] > myeps && idx[next] != start) {
                    // put next into q if next is not in q
                    idx[next] = start;//(int) q.size();
                    //q.push_back(next);
                    q[pointer_q++] = next;
                    pointer_q %= graph.n;
                }
            }
        }
    }
}

void multi_power_iter(const Graph &graph, const vector<int> &source,
                      unordered_map<int, vector<pair<int, double>>> &map_topk_ppr) {
    static thread_local unordered_map<int, double> map_ppr;
    for (int start: source) {
        fwd_power_iteration(graph, start, map_ppr);

        vector<pair<int, double>> temp_top_ppr(config.k);
        partial_sort_copy(map_ppr.begin(), map_ppr.end(), temp_top_ppr.begin(), temp_top_ppr.end(),
                          [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

        map_ppr.clear();
        map_topk_ppr[start] = temp_top_ppr;
        INFO(start);
    }
}

void multi_power_iter_self(const Graph &graph, const vector<int> &source,
                           unordered_map<int, double> &map_self_ppr) {
    //static thread_local unordered_map<int, double> map_ppr;
    static int count = 0;
    static thread_local Bwdidx bwd_idx_th;
    bwd_idx_th.first.initialize(graph.n);
    bwd_idx_th.second.initialize(graph.n);
    fill(bwd_idx_th.first.occur.m_data, bwd_idx_th.first.occur.m_data + graph.n, -1);
    fill(bwd_idx_th.second.occur.m_data, bwd_idx_th.second.occur.m_data + graph.n, -1);
    static thread_local vector<int> idx(graph.n, -1), q(graph.n);
    for (int start: source) {
        count++;
        if (count % 1000 == 0) INFO(count);
        fwd_power_iteration_self(graph, start, bwd_idx_th, idx, q);
        std::mutex g_mutex;
        /*
        g_mutex.lock();
        INFO(start);
        INFO(bwd_idx_th.first[start]);
        g_mutex.unlock();*/
        map_self_ppr[start] = bwd_idx_th.first[start];
    }
}

void gen_exact_self(const Graph &graph) {
    // config.epsilon = 0.5;
    // montecarlo_setting();
    load_exact_topk_ppr();
    map<int, double> ppr_self_old = load_exact_self_ppr();
    //load_exact_self_ppr();
    //map<int ,double> ppf_self=load_exact_self_ppr();
    //vector<double> ppf_self=load_exact_self_ppr_vec();
    /*
    unordered_map<int,double> ppr_self_old;
    for(auto item1:ppf_self){
        ppr_self_old[item1.first]=item1.second;
    }*/
    double min_rmax = 1.0 / 2000;
    set<int> candidate_node;
    bwd_idx.first.initialize(graph.n);
    bwd_idx.second.initialize(graph.n);
    vector<int> idx(graph.n, -1), node_with_r(graph.n), q(graph.n);
    int pointer_r = 0, max_candi_num = ceil(config.k / (2 - config.alpha) / config.alpha);
    int cur = 0;
    for (auto item:exact_topk_pprs) {
        fill(bwd_idx.first.occur.m_data, bwd_idx.first.occur.m_data + graph.n, -1);
        fill(bwd_idx.second.occur.m_data, bwd_idx.second.occur.m_data + graph.n, -1);
        if (++cur > config.query_size)break;
        int source_node = item.first;
        INFO(source_node);
        unordered_map<int, double> upper_bound, lower_bound;
        unordered_map<int, double> upper_bound_self, lower_bound_self;
        vector<int> candidate_s;
        for (int j = 0; j < max_candi_num; ++j) {
            int node = item.second[j].first;
            double ppr = item.second[j].second;
            config2.rmax = min_rmax;
            //reverse_local_update_linear(node, graph);
            reverse_local_update_linear_dht(node, graph, idx, node_with_r, pointer_r, q);
            upper_bound_self[node] = bwd_idx.first[node] + min_rmax;
            lower_bound_self[node] = bwd_idx.first[node];
            upper_bound[node] = ppr / bwd_idx.first[node];
            lower_bound[node] = ppr / (bwd_idx.first[node] + min_rmax);
            //INFO(node,upper_bound[node],lower_bound[node],ppr,bwd_idx.first[node]);
        }
        vector<double> tmp_dht(lower_bound.size());
        int cur = 0;
        for (auto item:lower_bound) {
            tmp_dht[cur++] = item.second;
        }
        nth_element(tmp_dht.begin(), tmp_dht.begin() + config.k - 1, tmp_dht.end(), cmp);
        cur = 0;
        //INFO(tmp_dht[config.k - 1]);
        for (auto item:upper_bound) {
            if (item.second >= tmp_dht[config.k - 1]) {
                if (ppr_self_old.find(item.first) == ppr_self_old.end() ||
                    ppr_self_old.at(item.first) > upper_bound_self[item.first] ||
                    ppr_self_old.at(item.first) < lower_bound_self[item.first]) {
                    cur++;
                    cout << item.first << "\t";
                    candidate_node.emplace(item.first);
                }
            }
        }
        INFO(cur);
    }
    vector<int> queries;
    queries.assign(candidate_node.begin(), candidate_node.end());
    unsigned int query_size = queries.size();
    INFO(queries.size());
    split_line();
    // montecarlo_setting();

    unsigned NUM_CORES = std::thread::hardware_concurrency() - 2;
    assert(NUM_CORES >= 2);

    int num_thread = min(query_size, NUM_CORES);
    int avg_queries_per_thread = query_size / num_thread;

    vector<vector<int>> source_for_all_core(num_thread);
    vector<unordered_map<int, double >> ppv_self_for_all_core(num_thread);

    for (int tid = 0; tid < num_thread; tid++) {
        int s = tid * avg_queries_per_thread;
        int t = s + avg_queries_per_thread;

        if (tid == num_thread - 1)
            t += query_size % num_thread;

        for (; s < t; s++) {
            // cout << s+1 <<". source node:" << queries[s] << endl;
            source_for_all_core[tid].push_back(queries[s]);
        }
    }


    {
        Timer timer(PI_QUERY_SELF);
        INFO("power itrating...");
        std::vector<std::future<void> > futures(num_thread);
        for (int tid = 0; tid < num_thread; tid++) {
            futures[tid] = std::async(std::launch::async, multi_power_iter_self, std::ref(graph),
                                      std::ref(source_for_all_core[tid]), std::ref(ppv_self_for_all_core[tid]));
        }
        std::for_each(futures.begin(), futures.end(), std::mem_fn(&std::future<void>::wait));
    }

    // cout << "average iter times:" << num_iter_topk/query_size << endl;
    cout << "average generation time (s): " << Timer::used(PI_QUERY_SELF) * 1.0 / query_size << endl;

    INFO("combine results...");
    //map<int, double> ppr_self;
    for (int tid = 0; tid < num_thread; tid++) {
        for (auto &ppv: ppv_self_for_all_core[tid]) {
            //exact_topk_pprs.insert(ppv);
            ppr_self_old[ppv.first] = ppv.second;
        }
    }
    save_self_ppr(ppr_self_old);
    //save_exact_topk_ppr();
}

set<int> gen_exact_topk(const Graph &graph) {
    // config.epsilon = 0.5;
    // montecarlo_setting();

    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min(query_size, config.query_size);
    INFO(query_size);

    assert(config.k < graph.n - 1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();
    // montecarlo_setting();

    unsigned NUM_CORES = std::thread::hardware_concurrency() - 1;
    assert(NUM_CORES >= 2);

    int num_thread = min(query_size, NUM_CORES);
    int avg_queries_per_thread = query_size / num_thread;

    vector<vector<int>> source_for_all_core(num_thread);
    vector<unordered_map<int, vector<pair<int, double>>>> ppv_for_all_core(num_thread);

    for (int tid = 0; tid < num_thread; tid++) {
        int s = tid * avg_queries_per_thread;
        int t = s + avg_queries_per_thread;

        if (tid == num_thread - 1)
            t += query_size % num_thread;

        for (; s < t; s++) {
            // cout << s+1 <<". source node:" << queries[s] << endl;
            source_for_all_core[tid].push_back(queries[s]);
        }
    }


    {
        Timer timer(PI_QUERY);
        INFO("power itrating...");
        std::vector<std::future<void> > futures(num_thread);
        for (int tid = 0; tid < num_thread; tid++) {
            futures[tid] = std::async(std::launch::async, multi_power_iter, std::ref(graph),
                                      std::ref(source_for_all_core[tid]), std::ref(ppv_for_all_core[tid]));
        }
        std::for_each(futures.begin(), futures.end(), std::mem_fn(&std::future<void>::wait));
    }

    // cout << "average iter times:" << num_iter_topk/query_size << endl;
    cout << "average generation time (s): " << Timer::used(PI_QUERY) * 1.0 / query_size << endl;

    INFO("combine results...");
    set<int> results;
    for (int tid = 0; tid < num_thread; tid++) {
        for (auto &ppv: ppv_for_all_core[tid]) {
            exact_topk_pprs.insert(ppv);
            for (auto item:ppv.second) {
                results.emplace(item.first);
            }
        }
        ppv_for_all_core[tid].clear();
    }

    save_exact_topk_ppr();
    return results;
}

void topk(Graph &graph) {
    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min(query_size, config.query_size);
    vector<int> true_queries;
    int used_counter = 0;

    assert(config.k < graph.n - 1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();
    // get exact topk dhts for each query node
    load_exact_topk_ppr();
    map<int, double> ppr_self = load_exact_self_ppr();

    bwd_idx.first.initialize(graph.n);
    bwd_idx.second.initialize(graph.n);
    int cur = 0;
    for (auto item:exact_topk_pprs) {
        if (++cur > config.query_size)break;
        int source_node = item.first;
        true_queries.emplace_back(source_node);
        unordered_map<int, double> dht;
        vector<pair<int, double>> topk_pprs = item.second;
        for (int k = 0; k < topk_pprs.size(); ++k) {
            int node = topk_pprs[k].first;
            if (ppr_self.find(node) != ppr_self.end()) {
                dht[node] = topk_pprs[k].second / ppr_self.at(node);
            }
        }
        vector<pair<int, double>> temp_top_dht(500);
        partial_sort_copy(dht.begin(), dht.end(), temp_top_dht.begin(), temp_top_dht.end(),
                          [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });
        exact_topk_dhts[source_node] = temp_top_dht;
    }

    if (config.algo == MC) {
        unsigned int step = config.k / 5;
        if (step > 0) {
            for (unsigned int i = 1; i < 5; i++) {
                ks.push_back(i * step);
            }
        }
        ks.push_back(config.k);
        for (auto k: ks) {
            PredResult rst(0, 0, 0, 0, 0);
            pred_results.insert(MP(k, rst));
        }
    }

    used_counter = 0;
    if (config.algo == MC) {
        rw_counter.initialize(graph.n);
        dht.initialize(graph.n);
        montecarlo_setting();
    } else if (config.algo == DNE) {
        dht.initialize(graph.n);
    } else if (config.algo == FORA_BIPPR) {
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        rw_counter.init_keys(graph.n);
        upper_bounds.init_keys(graph.n);
        lower_bounds.init_keys(graph.n);
        ppr.initialize(graph.n);
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);
        double raw_epsilon = config.epsilon, ratio = 1;
        fb_raw_setting(graph.n, graph.m, ratio, raw_epsilon);
    } else if (config.algo == FBRW) {
        fwd_idx.first.initialize(graph.n);//forward p
        fwd_idx.second.initialize(graph.n);//forward r
        rw_counter.init_keys(graph.n);//
        rw_bippr_counter.init_keys(graph.n);
        upper_bounds.init_keys(graph.n);
        lower_bounds.init_keys(graph.n);
        upper_bounds_self.init_keys(graph.n);
        upper_bounds_self_init.init_keys(graph.n);
        lower_bounds_self.init_keys(graph.n);
        upper_bounds_dht.init_keys(graph.n);
        lower_bounds_dht.init_keys(graph.n);
        ppr.initialize(graph.n);
        ppr_bi.initialize(graph.n);
        dht.initialize(graph.n);
        topk_filter.initialize(graph.n);
    }

    for (int i = 0; i < query_size; i++) {
        cout << i + 1 << ". source node:" << true_queries[i] << endl;
        get_topk(true_queries[i], graph);
        if (!config.NDCG) continue;
        if (config.algo == MC) {
            compute_NDCG_for_dif_k_dht(true_queries[i], ppr_self, graph);
        } else if (config.algo != DNE) {
            compute_NDCG(true_queries[i], ppr_self, graph);
        }
        split_line();
    }

    cout << "average iter times:" << num_iter_topk / query_size << endl;
    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);

    if (config.algo == MC) {
        display_precision_for_dif_k();
    }
}

void query_dht(Graph &graph) {
    INFO(config.algo);
    vector<int> queries;
    load_ss_query(queries);
    unsigned int query_size = queries.size();
    query_size = min(query_size, config.query_size);
    INFO(query_size);
    int used_counter = 0;

    assert(config.rmax_scale >= 0);
    INFO(config.rmax_scale);

    ppr.initialize(graph.n);
    dht.init_keys(graph.n);
    if (config.algo == MC) {
        montecarlo_setting();
        display_setting();
        used_counter = MC_DHT_QUERY;

        rw_counter.initialize(graph.n);

        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            montecarlo_query_dht(queries[i], graph);
            split_line();
        }
    } else if (config.algo == FORA_BIPPR) {
        used_counter = FBRAW_QUERY;
        double raw_epsilon = config.epsilon, ratio = 1;
        fb_raw_setting(graph.n, graph.m, ratio, raw_epsilon);
        display_setting();
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);

        rw_counter.initialize(graph.n);
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            fora_bippr_query(queries[i], graph);
            split_line();
        }

    } else if (config.algo == GI) {
        used_counter = GI_QUERY;
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            global_iteration_query(queries[i], graph);
            split_line();
        }
    } else if (config.algo == DNE) {
        used_counter = DNE_QUERY;
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            dne_query(queries[i], graph);
            split_line();
        }
    } else if (config.algo == FBRW) {
        double raw_epsilon = config.epsilon, ratio = 1;
        fora_bippr_setting(graph.n, graph.m, ratio, raw_epsilon);
        display_setting();
        used_counter = FB_QUERY;
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);

        rw_counter.initialize(graph.n);
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            fbrw_query(queries[i], graph, raw_epsilon);
            split_line();
        }
        cout << "num_total_fo" << num_total_fo << endl << "num_total_bi" << num_total_bi << endl;
    }
    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);
}


#endif //FORA_QUERY_H
