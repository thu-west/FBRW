//
// Created by 刘凯鑫 on 2019/12/5.
//

#ifndef FORA_HUBPPR_H
#define FORA_HUBPPR_H


class hubppr {
    void hubppr_query(int s, const Graph &graph) {
        Timer timer(HUBPPR_QUERY);

        ppr.reset_zero_values();

        {
            Timer tm(RONDOM_WALK);
            fwd_with_hub_oracle(graph, s);
            count_hub_dest();
            INFO("finish fwd work", hub_counter.occur.m_num, rw_counter.occur.m_num);
        }

        {
            Timer tm(BWD_LU);
            for (int t = 0; t < graph.n; t++) {
                bwd_with_hub_oracle(graph, t);
                // reverse_local_update_linear(t, graph);
                if ((bwd_idx.first.notexist(s) || 0 == bwd_idx.first[s]) && 0 == bwd_idx.second.occur.m_num) {
                    continue;
                }

                if (rw_counter.occur.m_num < bwd_idx.second.occur.m_num) { //iterate on smaller-size list
                    for (int i = 0; i < rw_counter.occur.m_num; i++) {
                        int node = rw_counter.occur[i];
                        if (bwd_idx.second.exist(node)) {
                            ppr[t] += bwd_idx.second[node] * rw_counter[node];
                        }
                    }
                } else {
                    for (int i = 0; i < bwd_idx.second.occur.m_num; i++) {
                        int node = bwd_idx.second.occur[i];
                        if (rw_counter.exist(node)) {
                            ppr[t] += rw_counter[node] * bwd_idx.second[node];
                        }
                    }
                }
                ppr[t] = ppr[t] / config.omega;
                if (bwd_idx.first.exist(s))
                    ppr[t] += bwd_idx.first[s];
            }
        }

#ifdef CHECK_PPR_VALUES
        display_ppr();
#endif
    }
};
void hubppr_query_topk_martingale(int s, const Graph &graph) {
    unsigned long long the_omega =
            2 * config.rmax * log(2 * config.k / config.pfail) / config.epsilon / config.epsilon / config.delta;
    static double bwd_cost_div = 1.0 * graph.m / graph.n / config.alpha;

    static double min_ppr = 1.0 / graph.n;
    static double new_pfail = config.pfail / 2.0 / graph.n / log2(1.0 * graph.n * config.alpha * graph.n * graph.n);
    static double pfail_star = log(new_pfail / 2);

    static std::vector<bool> target_flag(graph.n);
    static std::vector<double> m_omega(graph.n);
    static vector<vector<int>> node_targets(graph.n);
    static double cur_rmax = 1;

    // rw_counter.clean();
    for (int t = 0; t < graph.n; t++) {
        map_lower_bounds[t].second = 0;//min_ppr;
        upper_bounds[t] = 1.0;
        target_flag[t] = true;
        m_omega[t] = 0;
    }

    int num_iter = 1;
    int target_size = graph.n;
    if (cur_rmax > config.rmax) {
        cur_rmax = config.rmax;
        for (int t = 0; t < graph.n; t++) {
            if (target_flag[t] == false)
                continue;
            reverse_local_update_topk(s, t, reserve_maps[t], cur_rmax, residual_maps[t], graph);
            for (const auto &p: residual_maps[t]) {
                node_targets[p.first].push_back(t);
            }
        }
    }
    while (target_size > config.k &&
           num_iter <= 64) { //2^num_iter <= 2^64 since 2^64 is the largest unsigned integer here
        unsigned long long num_rw = pow(2, num_iter);
        rw_counter.clean();
        generate_accumulated_fwd_randwalk(s, graph, num_rw);
        updated_pprs.clean();
        // update m_omega
        {
            for (int x = 0; x < rw_counter.occur.m_num; x++) {
                int node = rw_counter.occur[x];
                for (const int t: node_targets[node]) {
                    if (target_flag[t] == false)
                        continue;
                    m_omega[t] += rw_counter[node] * residual_maps[t][node];
                    if (!updated_pprs.exist(t))
                        updated_pprs.insert(t, 1);
                }
            }
        }

        double b = (2 * num_rw - 1) * pow(cur_rmax / 2.0, 2);
        double lambda = sqrt(pow(cur_rmax * pfail_star / 3, 2) - 2 * b * pfail_star) - cur_rmax * pfail_star / 3;
        {
            for (int i = 0; i < updated_pprs.occur.m_num; i++) {
                int t = updated_pprs.occur[i];
                if (target_flag[t] == false)
                    continue;

                double reserve = 0;
                if (reserve_maps[t].find(s) != reserve_maps[t].end()) {
                    reserve = reserve_maps[t][s];
                }
                set_martingale_bound(lambda, 2 * num_rw - 1, t, reserve, cur_rmax, pfail_star, min_ppr, m_omega[t]);
            }
        }

        topk_pprs.clear();
        topk_pprs.resize(config.k);
        partial_sort_copy(map_lower_bounds.begin(), map_lower_bounds.end(), topk_pprs.begin(), topk_pprs.end(),
                          [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

        double k_bound = topk_pprs[config.k - 1].second;
        if (k_bound * (1 + config.epsilon) >= upper_bounds[topk_pprs[config.k - 1].first] ||
            (num_rw >= the_omega && cur_rmax <= config.rmax)) {
            break;
        }

        for (int t = 0; t < graph.n; t++) {
            if (target_flag[t] == true && upper_bounds[t] <= k_bound) {
                target_flag[t] = false;
                target_size--;
            }
        }
        num_iter++;
    }
}
void batch_topk(Graph &graph) {
    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min(query_size, config.query_size);
    int used_counter = 0;

    assert(config.k < graph.n - 1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();

    used_counter = 0;
    if (config.algo == FORA) {
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        rw_counter.init_keys(graph.n);
        upper_bounds.init_keys(graph.n);
        lower_bounds.init_keys(graph.n);
        ppr.initialize(graph.n);
        topk_filter.initialize(graph.n);
    } else if (config.algo == MC) {
        rw_counter.initialize(graph.n);
        ppr.initialize(graph.n);
        montecarlo_setting();
    } else if (config.algo == BIPPR) {
        bippr_setting(graph.n, graph.m);
        rw_counter.initialize(graph.n);
        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);
        ppr.initialize(graph.n);
    } else if (config.algo == FWDPUSH) {
        fwdpush_setting(graph.n, graph.m);
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        ppr.initialize(graph.n);
    } else if (config.algo == HUBPPR) {
        hubppr_topk_setting(graph.n, graph.m);
        rw_counter.initialize(graph.n);
        upper_bounds.init_keys(graph.n);
        if (config.with_rw_idx) {
            hub_counter.initialize(graph.n);
            load_hubppr_oracle(graph);
        }
        residual_maps.resize(graph.n);
        reserve_maps.resize(graph.n);
        map_lower_bounds.resize(graph.n);
        for (int v = 0; v < graph.n; v++) {
            residual_maps[v][v] = 1.0;
            map_lower_bounds[v] = MP(v, 0);
        }
        updated_pprs.initialize(graph.n);
    }

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

    // not FORA, so it's of single source
    // no need to change k to run again
    // check top-k results for different k
    if (config.algo != FORA && config.algo != HUBPPR) {
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            get_topk(queries[i], graph);
            split_line();
        }

        display_time_usage(used_counter, query_size);
        set_result(graph, used_counter, query_size);

        display_precision_for_dif_k();
    } else { // for FORA, when k is changed, run algo again
        for (unsigned int k: ks) {
            config.k = k;
            INFO("========================================");
            INFO("k is set to be ", config.k);
            result.topk_recall = 0;
            result.topk_precision = 0;
            result.real_topk_source_count = 0;
            Timer::clearAll();
            for (int i = 0; i < query_size; i++) {
                cout << i + 1 << ". source node:" << queries[i] << endl;
                get_topk(queries[i], graph);
                split_line();
            }
            pred_results[k].topk_precision = result.topk_precision;
            pred_results[k].topk_recall = result.topk_recall;
            pred_results[k].real_topk_source_count = result.real_topk_source_count;

            cout << "k=" << k << " precision=" << result.topk_precision / result.real_topk_source_count
                 << " recall=" << result.topk_recall / result.real_topk_source_count << endl;
            cout << "Average query time (s):" << Timer::used(used_counter) / query_size << endl;
            Timer::reset(used_counter);
        }

        // display_time_usage(used_counter, query_size);
        display_precision_for_dif_k();
    }
}
void bwd_with_hub_oracle(const Graph &graph, int t) {
    bwd_idx.first.clean();
    bwd_idx.second.clean();

    static unordered_map<int, bool> idx;
    idx.clear();

    // static vector<bool> idx(graph.n);
    // std::fill(idx.begin(), idx.end(), false);

    vector<int> q;
    q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;

    double myeps = config.rmax;
    // residual.clear();
    // exist.clear();

    q.push_back(t);
    // residual[t] = init_residual;
    bwd_idx.second.insert(t, 1);

    idx[t] = true;
    while (left < q.size()) {
        int v = q[left];
        idx[v] = false;
        left++;
        if (bwd_idx.second[v] < myeps)
            break;

        if (hub_bwd_idx.find(v) != hub_bwd_idx.end()) {
            vector<HubBwdidxWithResidual> &idxv = hub_bwd_idx[v];
            for (int i = idxv.size() - 1; i >= 0; i--) {
                HubBwdidxWithResidual &x = idxv[i];
                if (x.first >= bwd_idx.second[v]) {
                    HubBwdidx &useidx = x.second;
                    for (auto &residualkv:useidx.first) {
                        int next = residualkv.first;
                        double delta = residualkv.second * bwd_idx.second[v] / x.first;
                        if (bwd_idx.first.notexist(next)) {
                            bwd_idx.first.insert(next, delta);
                        } else {
                            bwd_idx.first[next] += delta;
                        }
                    }
                    if (useidx.second.size() <= 1) {
                        bwd_idx.second[v] = 0;
                        break;
                    }
                    for (auto &residualkv:useidx.second) {
                        int next = residualkv.first;
                        double delta = residualkv.second * bwd_idx.second[v] / x.first;

                        if (bwd_idx.second.notexist(next))
                            bwd_idx.second.insert(next, delta);
                        else
                            bwd_idx.second[next] += delta;

                        if (bwd_idx.second[next] >= myeps && idx[next] != true) {
                            // put next into q if next is not in q
                            idx[next] = true;//(int) q.size();
                            q.push_back(next);
                        }
                    }
                    bwd_idx.second[v] = 0;
                    break;
                }
            }
        } else {
            if (bwd_idx.first.notexist(v))
                bwd_idx.first.insert(v, bwd_idx.second[v] * config.alpha);
            else
                bwd_idx.first[v] += bwd_idx.second[v] * config.alpha;

            double residual = (1 - config.alpha) * bwd_idx.second[v];

            for (int next : graph.gr[v]) {
                int cnt = graph.g[next].size();
                // residual[next] += ((1 - config.alpha) * residual[v]) / cnt;
                if (bwd_idx.second.notexist(next))
                    bwd_idx.second.insert(next, residual / cnt);
                else
                    bwd_idx.second[next] += residual / cnt;

                if (bwd_idx.second[next] > myeps && idx[next] != true) {
                    // put next into q if next is not in q
                    idx[next] = true;//(int) q.size();
                    q.push_back(next);
                }
            }
            bwd_idx.second[v] = 0;
        }

    }
}
void fwd_with_hub_oracle(const Graph &graph, int start) {
    num_total_rw += config.omega;

    rw_counter.clean();
    hub_counter.clean();

    unsigned long long num_tries = int64(config.omega / graph.n * hub_sample_number[start]);
    // INFO(num_tries, config.omega);

    if (graph.g[start].size() == 0) {
        if (rw_counter.notexist(start))
            rw_counter.insert(start, num_tries);
        else
            rw_counter[start] += num_tries;
        return;
    }

    if (hub_fwd_idx_size[start] != 0) {
        if (num_tries <= hub_fwd_idx_size[start]) {
            hub_counter.insert(start, num_tries);
            num_tries = 0;
        } else {
            hub_counter.insert(start, hub_fwd_idx_size[start]);
            num_tries -= hub_fwd_idx_size[start];
        }
    } else {
        if (rw_counter.notexist(start))
            rw_counter.insert(start, num_tries * config.alpha);
        else
            rw_counter[start] += num_tries * config.alpha;
        num_tries = num_tries * (1 - config.alpha);
    }

    for (int64 i = 0; i < num_tries; i++) {
        if (hub_fwd_idx_size[start] != 0) {
            int l = random_walk_with_compressed_forward_oracle(start, graph);
            if (l >= 0) {
                if (rw_counter.notexist(l)) {
                    rw_counter.insert(l, 1);
                } else {
                    rw_counter[l] += 1;
                }
            }
        } else {
            int random_out_neighbor = drand() * (graph.g[start].size() - 1);
            random_out_neighbor = graph.g[start][random_out_neighbor];
            int l = random_walk_with_compressed_forward_oracle(random_out_neighbor, graph);
            if (l >= 0) {
                if (rw_counter.notexist(l)) {
                    rw_counter.insert(l, 1);
                } else {
                    rw_counter[l] += 1;
                }
            }
        }
    }
}
static void reverse_local_update_heap(int t, const Graph &graph, double init_residual = 1) {
    static BinaryHeap<double, greater<double> > heap(graph.n, greater<double>());

    bwd_idx.first.clean();
    bwd_idx.second.clean();

    double myeps = config.rmax;

    heap.clear();
    heap.insert(t, init_residual);

    while (heap.size()) {
        auto top = heap.extract_top();
        double residual = top.first;
        int v = top.second;
        if (residual < myeps)
            break;

        heap.delete_top();
        if (bwd_idx.first.notexist(v)) {
            bwd_idx.first.insert(v, residual * config.alpha);
        } else {
            bwd_idx.first[v] += residual * config.alpha;
        }
        double resi = (1 - config.alpha) * residual;
        for (int next : graph.gr[v]) {
            int cnt = graph.g[next].size();
            double delta = resi / cnt;
            if (heap.has_idx(next))
                heap.modify(next, heap.get_value(next) + delta);
            else
                heap.insert(next, delta);
        }
    }

    for (auto item: heap.get_elements()) {
        bwd_idx.second.insert(item.second, item.first);
    }
}
void montecarlo_query2(int v, const Graph &graph) {
    Timer timer(MC_QUERY2);

    double fwd_rw_count = 3 * log(2 / config.pfail) / config.epsilon / config.epsilon / config.alpha;
    rw_counter.clean();
    ppr.reset_zero_values();

    {
        Timer tm(RONDOM_WALK2);
        num_total_rw += fwd_rw_count;
        for (unsigned long i = 0; i < fwd_rw_count; i++) {
            int destination = random_walk(v, graph);
            if (!rw_counter.exist(destination))
                rw_counter.insert(destination, 1);
            else
                rw_counter[destination] += 1;
        }
    }

    int node_id;
    for (long i = 0; i < rw_counter.occur.m_num; i++) {
        node_id = rw_counter.occur[i];
        ppr[node_id] = rw_counter[node_id] * 1.0 / fwd_rw_count;
    }

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

void compute_ppr_with_bwdidx_with_bound(const Graph &graph, double old_omega, double threshold) {
    INFO(config2.omega);
    ppr_bi.clean();
    for (int k = 0; k < dht.occur.m_num; ++k) {
        int nodeid = dht.occur[k];
        if (dht.notexist(nodeid)) { continue; }
        {
            Timer tm(RONDOM_WALK);
            num_total_rw += config2.omega;//单纯计数用
            for (unsigned long i = 0; i < config2.omega - old_omega; i++) {
                int destination = random_walk(nodeid, graph);
                multi_bwd_idx_rw[nodeid][destination] += 1;
            }
        }
        if (multi_bwd_idx_p[nodeid].find(nodeid) != multi_bwd_idx_p[nodeid].end()) {
            ppr_bi.insert(nodeid, multi_bwd_idx_p[nodeid].at(nodeid));
        } else {
            ppr_bi.insert(nodeid, config2.alpha);
        }
        if (config2.rmax < 1.0) {
            for (auto tmp:multi_bwd_idx_r[nodeid]) {
                int other_node = tmp.first;
                double residual = tmp.second;
                if (multi_bwd_idx_rw[nodeid].find(other_node) != multi_bwd_idx_rw[nodeid].end()) {
                    ppr_bi[nodeid] += multi_bwd_idx_rw[nodeid].at(other_node) / config.omega * residual;
                }
            }
        } else if (multi_bwd_idx_rw[nodeid].find(nodeid) != multi_bwd_idx_rw[nodeid].end()) {
            ppr_bi.insert(nodeid, multi_bwd_idx_rw[nodeid].at(nodeid) / config.omega);
        }
        double old_ub = upper_bounds_self[nodeid], old_lb = lower_bounds_self[nodeid];
        upper_bounds_self[nodeid] = min(upper_bounds_self_init[nodeid], ppr_bi[nodeid] * (1 + config2.epsilon));
        lower_bounds_self[nodeid] = max(config2.alpha, ppr_bi[nodeid] * (1 - config2.epsilon));
        assert(old_ub > upper_bounds_self[nodeid] && old_lb < lower_bounds_self[nodeid]);
    }
}
#endif //FORA_HUBPPR_H
