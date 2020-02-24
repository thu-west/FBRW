//Contributors: Sibo Wang, Renchi Yang
#ifndef __ALGO_H__
#define __ALGO_H__

#include "graph.h"
#include "heap.h"
#include "config.h"
#include <tuple>
// #include "sfmt/SFMT.h"
#include <queue>

using namespace std;


struct PredResult {
    double topk_avg_relative_err;
    double topk_avg_abs_err;
    double topk_recall;
    double topk_precision;
    int real_topk_source_count;
    double topk_NDCG;
    PredResult(double mae = 0, double mre = 0, double rec = 0, double pre = 0, int count = 0,double NDCG=0) :
            topk_avg_relative_err(mae),
            topk_avg_abs_err(mre),
            topk_recall(rec),
            topk_precision(pre),
            real_topk_source_count(count),
            topk_NDCG(NDCG){}
};

unordered_map<int, PredResult> pred_results;

Fwdidx fwd_idx;
Bwdidx bwd_idx;
iMap<double> ppr;

iMap<double> ppr_bi;
iMap<double> dht;

iMap<int> topk_filter;
// vector< boost::atomic<double> > vec_ppr;
iMap<int> rw_counter;

iMap<double> rw_bippr_counter;
// RwIdx rw_idx;
atomic<unsigned long long> num_hit_idx;
atomic<unsigned long long> num_total_rw;
atomic<unsigned long long> num_total_bi;
atomic<unsigned long long> num_total_fo;
long num_iter_topk;
vector<int> rw_idx;
vector<pair<unsigned long long, unsigned long> > rw_idx_info;
unordered_map<int, vector<int>> rw_index;

map<int, vector<pair<int, double> > > exact_topk_pprs;
map<int, vector<pair<int, double>>> exact_topk_dhts;
vector<pair<int, double> > topk_pprs;
vector<pair<int, double> > topk_dhts;

iMap<double> upper_bounds;
iMap<double> lower_bounds;

iMap<double> upper_bounds_self;
iMap<double> upper_bounds_self_init;
iMap<double> lower_bounds_self;

iMap<double> upper_bounds_dht;
iMap<double> lower_bounds_dht;

unordered_map<int, double> multi_bwd_idx_p;
unordered_map<int, unordered_map<int, double>> multi_bwd_idx_r;


vector<pair<int, double>> map_lower_bounds;

// for hubppr
vector<int> hub_fwd_idx;
//pointers to compressed fwd_idx, nodeid:{ start-pointer, start-pointer, start-pointer,...,end-pointer }
map<int, vector<unsigned long long> > hub_fwd_idx_cp_pointers;
vector<vector<unsigned long long>> hub_fwd_idx_ptrs;
vector<int> hub_fwd_idx_size;
iMap<int> hub_fwd_idx_size_k;
vector<int> hub_sample_number;
iMap<int> hub_counter;

map<int, vector<HubBwdidxWithResidual>> hub_bwd_idx;

unsigned concurrency;

vector<int> ks;

vector<unordered_map<int, double>> residual_maps;
vector<map<int, double>> reserve_maps;

inline uint32_t xor128(void) {
    static uint32_t x = 123456789;
    static uint32_t y = 362436069;
    static uint32_t z = 521288629;
    static uint32_t w = 88675123;
    uint32_t t;
    t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}

inline static unsigned long xshift_lrand() {
    return (unsigned long) xor128();
}

inline static double xshift_drand() {
    return ((double) xshift_lrand() / (double) UINT_MAX);
}

inline static unsigned long lrand() {
    return rand();
    // return sfmt_genrand_uint32(&sfmtSeed);
}

inline static double drand() {
    return rand() * 1.0f / RAND_MAX;
    // return sfmt_genrand_real1(&sfmtSeed);
}

inline int random_walk(int start, const Graph &graph) {
    int cur = start;
    unsigned long k;
    if (graph.g[start].size() == 0) {
        return start;
    }
    while (true) {
        if (drand() < config.alpha) {
            return cur;
        }
        if (graph.g[cur].size()) {
            k = lrand() % graph.g[cur].size();
            cur = graph.g[cur][k];
        } else {
            cur = start;
        }
    }
}

inline void random_walk_dht(int start, int pathid, const Graph &graph, unordered_map<int, pair<int, int>> &occur) {
    int cur = start;
    unsigned long k;
    bool flag = true;
    if (graph.g[start].size() == 0) {
        //return start;
        flag = false;
        if (occur.find(cur) == occur.end()) {
            occur.emplace(cur, make_pair(pathid, 1));
        } else if (occur.at(cur).first != pathid) {
            occur.at(cur).first = pathid;
            occur.at(cur).second++;
        }
    }
    while (flag) {
        if (occur.find(cur) == occur.end()) {
            occur.emplace(cur, make_pair(pathid, 1));
        } else if (occur.at(cur).first != pathid) {
            occur.at(cur).first = pathid;
            occur.at(cur).second++;
        }
        if (drand() < config.alpha) {
            //return cur;
            break;
        }
        if (graph.g[cur].size()) {
            k = lrand() % graph.g[cur].size();
            cur = graph.g[cur][k];
        } else {
            cur = start;
        }
    }
}

void global_iteration(int v, vector<double> &dht_old, const Graph &graph) {
    vector<double> new_dht;
    /*
    for(auto item:dht_old){
        int nodeid=item.first;
        for(int nei:graph.gr[nodeid]){
            if (nei==v)continue;
            int deg=graph.g[nei].size();
            new_dht[nei] += (1-config.alpha)/deg* item.second;
        }
    }*/
    double max_dht = 0;
    for (int i = 0; i < graph.n; ++i) {
        if (i == v)continue;
        int deg = graph.g[i].size();
        for (int nei:graph.g[i]) {
            new_dht[i] += (1 - config.alpha) / deg * dht_old.at(nei);
        }
    }
    new_dht[v] += 1;
    swap(dht_old, new_dht);
}

inline double compute_dne_error(double max_error, int k, double max_gap_error) {

    double lambda = pow((1 - config.alpha), k) / config.alpha * max_gap_error;
    return (1 - config.alpha) * (1 - config.alpha) / (config.alpha * (2 - config.alpha)) * (max_error + lambda) +
           lambda;
}

inline bool if_dne_stop(double &old_error, double max_error, int k, double max_gap_error) {
    double new_error = compute_dne_error(max_error, k, max_gap_error);
    cout << new_error << " " << old_error << " " << config.epsilon * config.delta << endl;
    if (new_error <= config.epsilon * config.delta) return true;
    if (old_error >= new_error && old_error / new_error < 1.0 + pow(0.1, 10)) return true;
    old_error = new_error;
    return false;
}

double dhe_query_basic(int query_node, int v, int den_m, const Graph &graph) {
    unordered_map<int, double> dht_to_v, dht_to_v_copy;
    dht_to_v[v] = 1;
    set<int> neighbors, boundary;
    neighbors.emplace(v);
    boundary.emplace(v);
    int max_node = -1;
    double max_error = 0, factor_m =
            (1 - config.alpha) * (1 - config.alpha) / (config.alpha * (2 - config.alpha));//\tau^2/(1-\tau^2)
    // expand
    while (neighbors.size() < den_m && !boundary.empty()) {
        max_error = 0;
        for (int node_in_bound:boundary) {
            if (dht_to_v.at(node_in_bound) > max_error) {
                max_error = dht_to_v.at(node_in_bound);
                max_node = node_in_bound;
            }
        }
        if (max_error * factor_m < config.epsilon * config.delta) {
            break;
        }
        boundary.erase(max_node);
        for (int nei:graph.gr[max_node]) {
            neighbors.emplace(nei);
        }
        for (int nei:graph.gr[max_node]) {
            for (int nei_of_nei:graph.gr[nei]) {
                if (neighbors.find(nei_of_nei) == neighbors.end()) {
                    boundary.emplace(nei);
                    break;
                }
            }
        }
        for (int node:neighbors) {
            if (node == v) {
                dht_to_v[node] = 1;
                continue;
            } else {
                dht_to_v[node] = 0;
            }
            int deg = graph.g[node].size();
            for (int nei:graph.g[node]) {
                dht_to_v[node] += (1 - config.alpha) / deg * dht_to_v[nei];
            }
        }
    }
    //refinement
    Timer tm(111);
    max_error = 0;
    int k = 0;
    double old_error = 0, max_gap_first = 0;
    do {
        k++;
        for (int node:neighbors) {
            if (node == v) {
                dht_to_v_copy[node] = 1;
                continue;
            } else {
                dht_to_v_copy[node] = 0;
            }
            int deg = graph.g[node].size();
            for (int nei:graph.g[node]) {
                dht_to_v_copy[node] += (1 - config.alpha) / deg * dht_to_v[nei];
            }
            if (boundary.find(node) != boundary.end() && max_error < dht_to_v_copy[node]) {
                max_error = dht_to_v_copy[node];
            }
        }
        if (k == 1) {
            for (auto item:dht_to_v_copy) {
                if (max_gap_first < item.second - dht_to_v[item.first]) {
                    max_gap_first = item.second - dht_to_v[item.first];
                }
            }
        }
        swap(dht_to_v, dht_to_v_copy);
    } while (!if_dne_stop(old_error, max_error, k, max_gap_first));
    return dht_to_v[query_node];
}


unsigned int SEED = 1;

inline static unsigned long lrand_thd(int core_id) {
    //static thread_local std::mt19937 gen(core_id+1);
    //static std::uniform_int_distribution<> dis(0, INT_MAX);
    //return dis(gen);
    return rand_r(&SEED);
}

inline static double drand_thd(int core_id) {
    return ((double) lrand_thd(core_id) / (double) INT_MAX);
}

inline int random_walk_thd(int start, const Graph &graph, int core_id) {
    int cur = start;
    unsigned long k;
    if (graph.g[start].size() == 0) {
        return start;
    }
    while (true) {
        if (drand_thd(core_id) < config.alpha) {
            return cur;
        }
        if (graph.g[cur].size()) {
            k = lrand_thd(core_id) % graph.g[cur].size();
            cur = graph.g[cur][k];
        } else {
            cur = start;
        }
    }
}

void count_hub_dest() {
    // {   Timer tm(101);
    int remaining;
    unsigned long long last_beg_ptr;
    unsigned long long end_ptr;
    int hub_node;
    int blocked_num;
    int bit_pos;
    for (int i = 0; i < hub_counter.occur.m_num; i++) {
        hub_node = hub_counter.occur[i];
        last_beg_ptr = hub_fwd_idx_ptrs[hub_node][hub_fwd_idx_ptrs[hub_node].size() - 2];
        end_ptr = hub_fwd_idx_ptrs[hub_node][hub_fwd_idx_ptrs[hub_node].size() - 1];

        remaining = hub_counter[hub_node];

        if (remaining > hub_fwd_idx_size_k[hub_node]) {
            for (unsigned long long ptr = last_beg_ptr; ptr < end_ptr; ptr += 2) {
                if (rw_counter.notexist(hub_fwd_idx[ptr])) {
                    rw_counter.insert(hub_fwd_idx[ptr], hub_fwd_idx[ptr + 1]);
                } else {
                    rw_counter[hub_fwd_idx[ptr]] += hub_fwd_idx[ptr + 1];
                }
                remaining -= hub_fwd_idx[ptr + 1];
            }
        }


        for (int j = 0; j < hub_fwd_idx_ptrs[hub_node].size() - 2; j++) {
            bit_pos = 1 << j;
            if (bit_pos & remaining) {
                for (unsigned long long ptr = hub_fwd_idx_ptrs[hub_node][j];
                     ptr < hub_fwd_idx_ptrs[hub_node][j + 1]; ptr += 2) {
                    if (rw_counter.notexist(hub_fwd_idx[ptr])) {
                        rw_counter.insert(hub_fwd_idx[ptr], hub_fwd_idx[ptr + 1]);
                    } else {
                        rw_counter[hub_fwd_idx[ptr]] += hub_fwd_idx[ptr + 1];
                    }
                }
            }
        }
    }
    // }
}

inline int random_walk_with_compressed_forward_oracle(int start, const Graph &graph) {
    int cur = start;
    unsigned long k;
    if (graph.g[start].size() == 0) {
        return start;
    }
    while (true) {
        if (hub_fwd_idx_size[cur] != 0 && (hub_counter.notexist(cur) || hub_counter[cur] < hub_fwd_idx_size[cur])) {
            if (hub_counter.notexist(cur))
                hub_counter.insert(cur, 1);
            else
                hub_counter[cur] += 1;
            return -1;
        }

        if (drand() < config.alpha) {
            return cur;
        }

        if (graph.g[cur].size()) {
            k = lrand() % graph.g[cur].size();
            cur = graph.g[cur][k];
        } else
            cur = start;
    }
}

inline void split_line() {
    INFO("-----------------------------");
}

inline void display_setting() {
    INFO(config.epsilon);
    INFO(config.delta);
    INFO(config.pfail);
    INFO(config.rmax);
    INFO(config.omega);
    if (config2.pfail == config.pfail) {
        INFO(config2.epsilon);
        INFO(config2.delta);
        INFO(config2.pfail);
        INFO(config2.rmax);
        INFO(config2.omega);
    }
}


inline void display_ppr() {
    for (int i = 0; i < ppr.occur.m_num; i++) {
        cout << ppr.occur[i] << "->" << ppr[ppr.occur[i]] << endl;
    }
}

inline void display_dht() {
    for (int i = 0; i < dht.occur.m_num; i++) {
        if (dht[dht.occur[i]] <= 0)continue;
        cout << dht.occur[i] << "->" << dht[dht.occur[i]] << endl;
    }
}

static void display_time_usage(int used_counter, int query_size) {
    if (config.algo == FORA) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(RONDOM_WALK) * 100.0 / Timer::used(used_counter) << "%" << " for random walk cost" << endl;
        cout << Timer::used(FWD_LU) * 100.0 / Timer::used(used_counter) << "%" << " for forward push cost" << endl;
        // if(config.action == TOPK)
        // cout <<  Timer::used(SORT_MAP)*100.0/Timer::used(used_counter) << "%" << " for sorting top k cost" << endl;
        split_line();
    } else if (config.algo == FORA_MC) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(RONDOM_WALK) * 100.0 / Timer::used(used_counter) << "%" << " for random walk cost" << endl;
        cout << Timer::used(FWD_LU) * 100.0 / Timer::used(used_counter) << "%" << " for forward push cost" << endl;
        cout << Timer::used(MC_QUERY2) * 100.0 / Timer::used(used_counter) << "%" << " for montecarlo cost" << endl;
        split_line();
    } else if (config.algo == BIPPR) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(RONDOM_WALK) * 100.0 / Timer::used(used_counter) << "%" << " for random walk cost" << endl;
        cout << Timer::used(BWD_LU) * 100.0 / Timer::used(used_counter) << "%" << " for backward push cost" << endl;
    } else if (config.algo == MC) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(RONDOM_WALK) * 100.0 / Timer::used(used_counter) << "%" << " for random walk cost" << endl;
    } else if (config.algo == FWDPUSH) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(FWD_LU) * 100.0 / Timer::used(used_counter) << "%" << " for forward push cost" << endl;
    } else if (config.algo == MC) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(RONDOM_WALK) * 100.0 / Timer::used(used_counter) << "%" << " for random walk cost" << endl;
    }

    if (config.with_rw_idx)
        cout << "Average rand-walk idx hit ratio: " << num_hit_idx * 100.0 / num_total_rw << "%" << endl;

    if (config.action == TOPK) {
        assert(result.real_topk_source_count > 0);
        cout << "Average top-K Precision: " << result.topk_precision / result.real_topk_source_count << endl;
        cout << "Average top-K Recall: " << result.topk_recall / result.real_topk_source_count << endl;
    }

    cout << "Average query time (s):" << Timer::used(used_counter) / query_size << endl;
    cout << "Memory usage (MB):" << get_proc_memory() / 1000.0 << endl << endl;
}

static void set_result(const Graph &graph, int used_counter, int query_size) {
    config.query_size = query_size;

    result.m = graph.m;
    result.n = graph.n;
    result.avg_query_time = Timer::used(used_counter) / query_size;

    result.total_mem_usage = get_proc_memory() / 1000.0;
    result.total_time_usage = Timer::used(used_counter);

    result.num_randwalk = num_total_rw;

    if (config.with_rw_idx) {
        result.num_rw_idx_use = num_hit_idx;
        result.hit_idx_ratio = num_hit_idx * 1.0 / num_total_rw;
    }

    result.randwalk_time = Timer::used(RONDOM_WALK);
    result.randwalk_time_ratio = Timer::used(RONDOM_WALK) * 100 / Timer::used(used_counter);

    if (config.algo == FBRW) {
        result.propagation_time = Timer::used(FWD_LU) + Timer::used(BWD_LU);
        result.propagation_time_ratio = result.propagation_time * 100 / Timer::used(used_counter);
    }

    if (config.action == TOPK) {
        result.topk_sort_time = Timer::used(SORT_MAP);
        // result.topk_precision = avg_topk_precision;
        // result.topk_sort_time_ratio = Timer::used(SORT_MAP)*100/Timer::used(used_counter);
    }
}

inline void fora_setting(int n, long long m) {
    config.rmax = config.epsilon * sqrt(config.delta / 3 / m / log(2 / config.pfail));
    config.rmax *= config.rmax_scale;
    // config.rmax *= config.multithread_param;
    config.omega = (2 + config.epsilon) * log(2 / config.pfail) / config.delta / config.epsilon / config.epsilon;
}

inline void fb_raw_setting(int n, long long m, double ratio, double raw_epsilon) {
    config.pfail = 1.0 / 2 / n;
    config2 = config;

    config.delta = config.alpha / n;
    config2.delta = config2.alpha;
    config.epsilon = raw_epsilon * ratio / (ratio + 1 + raw_epsilon);
    config2.epsilon = raw_epsilon / (ratio + 1 + raw_epsilon);

    config.rmax = config.epsilon * sqrt(config.delta / 3 / m / log(2 / config.pfail));
    config.rmax *= config.rmax_scale;
    config.omega = (2 + config.epsilon) * log(2 / config.pfail) / config.delta / config.epsilon / config.epsilon;

    //config2.rmax = config2.epsilon * sqrt(config2.delta / 3 / log(2.0 / config2.pfail));
    config2.rmax = config2.epsilon *
                   sqrt(config2.delta * m / n / config2.alpha / (2 + config2.epsilon) / log(2.0 / config2.pfail));
    config2.rmax *= config2.rmax_scale;
    config2.omega = config2.rmax * (2 + config2.epsilon) * log(2.0 / config2.pfail) / config2.delta / config2.epsilon /
                    config2.epsilon;
}

bool compare_fora_bippr_cost(int k, int m = 0, double rsum = 0) {
    if (k == 0)
        return false;
    static double old_config2_rmax;
    //比较前向和后向的复杂度，如果前向高于后向，返回真
    double ratio;
    double cost_fora =
            (2 * config.epsilon / 3 + 2) * log(2 / config.pfail) / config.delta / config.epsilon / config.epsilon;
    INFO(cost_fora);
    INFO(1 / config.rmax + rsum * config.omega);
    INFO(k * (2 / config2.rmax));
    //ratio=(1/config.rmax+rsum*config.omega)/k/(2/config2.rmax);
    ratio = ((config.rmax * m + rsum) * cost_fora * (sqrt(2) - 1) / k / (2 / config2.rmax));
    old_config2_rmax = config2.rmax;
    return ratio > 1;
}

bool test_run_fora(int candidate_size, long m, int n, double forward_counter, double real_num_rw) {
    if (candidate_size == 0)
        return true;
    double new_rmax2 = config2.epsilon / 2 * sqrt(config2.delta * m / n / config2.alpha / (2 + config2.epsilon / 2) /
                                                  log(2.0 / config2.pfail));
    double cost_new = forward_counter * 1.4 + real_num_rw * 2;
    double cost_bi = candidate_size * 2 / new_rmax2 * m / n / config2.alpha;
    double cost_bi_old = candidate_size / config2.rmax * m / n / config2.alpha;
    //INFO(cost_new, candidate_size * 2 / new_rmax2, cost_bi);
    //INFO(2 / new_rmax2 * m / n / config2.alpha, new_rmax2);
    if (ppr_bi.occur.m_num == 0) {
        return cost_new < cost_bi;
    } else {
        return cost_new < cost_bi - cost_bi_old;
    }
}

inline void fora_bippr_setting(int n, long long m, double ratio, double raw_epsilon, bool topk = false) {
    if (!topk) {
        config.pfail = 1.0 / 2 / n;
        config.delta = config.alpha / n;
        config2 = config;

        config.epsilon = raw_epsilon * ratio / (ratio + 1 + raw_epsilon);
        config2.epsilon = raw_epsilon / (ratio + 1 + raw_epsilon);

        config2.delta = config2.alpha;
    }

    config.rmax = config.epsilon * sqrt(config.delta / 3 / m / log(2 / config.pfail));
    config.rmax *= config.rmax_scale;
    // config.rmax *= config.multithread_param;
    config.omega = (2 + config.epsilon) * log(2 / config.pfail) / config.delta / config.epsilon / config.epsilon;


    config2.rmax = config2.epsilon *
                   sqrt(config2.delta * m / n / config2.alpha / (2 + config2.epsilon) / log(2.0 / config2.pfail));
    config2.rmax *= config2.rmax_scale;
    config2.omega = config2.rmax * (2 + config2.epsilon) * log(2.0 / config2.pfail) / config2.delta / config2.epsilon /
                    config2.epsilon;
}

inline void montecarlo_setting() {
    double fwd_rw_count = 3 * log(2 / config.pfail) / config.epsilon / config.epsilon / config.delta;
    config.omega = fwd_rw_count;
}

inline bool check_cost(double rsum, double &ratio, double n, double m, long forward_push_num, double raw_epsilon) {

    int counter = 0;
    for (int j = 0; j < fwd_idx.first.occur.m_num; ++j) {
        int node = fwd_idx.first.occur[j];
        if (fwd_idx.first[node] > config.delta) {
            counter++;
        }
    }
    double cost_bippr = counter * (config2.omega + m / n / config.alpha / config2.rmax);//config.alpha;
    double cost_fora = rsum * config.omega + forward_push_num;
    double old_rmax = config.rmax, new_ratio = ratio / 2;
    int iter=0;
    do {
        iter++;
        // reduce ratio to minimize cost
        fora_bippr_setting(n, m, new_ratio, raw_epsilon);
        double cost_bippr_new = counter * (config2.omega + m / n / config.alpha / config2.rmax);//config.alpha;
        double cost_fora_new = rsum * config.omega + forward_push_num * old_rmax / config.rmax;

        if (cost_bippr + cost_fora > cost_bippr_new + cost_fora_new) {
            ratio = new_ratio;
            if (iter>1) return false;
            return true;
        } else {
            new_ratio = (new_ratio+ratio)/2;
        }
    } while (iter<4);
    fora_bippr_setting(n, m, ratio, raw_epsilon);
    return false;
}

inline void generate_ss_query(int n) {
    string filename = config.graph_location + "ssquery.txt";
    ofstream queryfile(filename);
    for (int i = 0; i < config.query_size; i++) {
        int v = rand() % n;
        queryfile << v << endl;
    }
}

void load_ss_query(vector<int> &queries) {
    string filename = config.query_high_degree ? config.graph_location + "high_degree_ssquery.txt" :
                      config.graph_location + "ssquery.txt";
    if (!file_exists_test(filename)) {
        cerr << "query file does not exist, please generate ss query files first" << endl;
        exit(0);
    }
    ifstream queryfile(filename);
    int v;
    while (queryfile >> v) {
        queries.push_back(v);
    }
}

void compute_precision_dht(int v) {
    double precision = 0.0;
    double recall = 0.0;

    if (exact_topk_dhts.size() > 1 && exact_topk_dhts.find(v) != exact_topk_dhts.end()) {

        unordered_map<int, double> topk_map;
        for (auto &p: topk_dhts) {
            if (p.second > 0) {
                topk_map.insert(p);
            }
        }

        unordered_map<int, double> exact_map;
        int size_e = min(config.k, (unsigned int) exact_topk_dhts[v].size());

        for (int i = 0; i < size_e; i++) {
            pair<int, double> &p = exact_topk_dhts[v][i];
            if (p.second > 0) {
                exact_map.insert(p);
                if (topk_map.find(p.first) != topk_map.end())
                    recall++;
            }
        }

        for (auto &p: topk_map) {
            if (exact_map.find(p.first) != exact_map.end()) {
                precision++;
            }
        }

        // for(int i=0; i<config.k; i++){
        //     cout << "NO." << i << " pred:" << topk_pprs[i].first << ", " << topk_pprs[i].second << "\t exact:" << exact_topk_dhts[v][i].first << ", " << exact_topk_dhts[v][i].second << endl;
        // }

        assert(exact_map.size() > 0);
        assert(topk_map.size() > 0);
        if (exact_map.size() <= 1)
            return;

        recall = recall * 1.0 / exact_map.size();
        precision = precision * 1.0 / exact_map.size();
        INFO(exact_map.size(), recall, precision);
        result.topk_recall += recall;
        result.topk_precision += precision;

        result.real_topk_source_count++;
    }
}

double backward_push(int s, int t, const Graph &graph, iMap<double> &r, double rmax, double init_residual = 1) {
    double p = 0;
    r.initialize(graph.n);
    r.clean();
    unordered_map<int, bool> idx;
    idx.clear();

    vector<int> q;
    q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;

    double myeps = rmax;

    q.push_back(t);
    r.insert(t, init_residual);

    idx[t] = true;
    while (left < q.size()) {
        int v = q[left];
        idx[v] = false;
        left++;
        if (r[v] < myeps)
            break;
        if (v == s) {
            p += r[v] * config.alpha;
        }

        double residual = (1 - config.alpha) * r[v];
        r[v] = 0;
        if (graph.gr[v].size() > 0) {
            for (int next : graph.gr[v]) {
                int cnt = graph.g[next].size();
                if (r.notexist(next))
                    r.insert(next, residual / cnt);
                else
                    r[next] += residual / cnt;

                if (r[next] > myeps && idx[next] != true) {
                    // put next into q if next is not in q
                    idx[next] = true;//(int) q.size();
                    q.push_back(next);
                }
            }
        }
    }
    return p;
}

double pair_wise_ppr(int source, int target, Graph &graph, double delta = 0.0000000001, double epsilon = 0.5) {
    double result = 0, pfail = 1.0 / graph.n;
    //set parameter
    double rmax = epsilon * sqrt(graph.m * delta / graph.n / (2 + epsilon) / log(2 / pfail));
    double omega = ceil(rmax * (2 + epsilon) * log(2 / pfail) / delta / epsilon / epsilon);
    iMap<double> r;
    result = backward_push(source, target, graph, r, rmax);
    for (unsigned long i = 0; i < omega; i++) {
        int destination = random_walk(source, graph);
        if (r.exist(destination)) {
            result += r[destination] / omega;
        }
    }
    return result;
}

void compute_NDCG(int v, const map<int, double> ppr_self, Graph &graph) {

    if (exact_topk_dhts.size() > 1 && exact_topk_dhts.find(v) != exact_topk_dhts.end()) {
        // first compute IDCG
        double iDCG = 0, DCG = 0, min_ssppr = 0;
        unordered_map<int, double> exact_map, exact_ppr_map;
        for (int k = 0; k < exact_topk_pprs[v].size(); ++k) {
            pair<int, double> &p = exact_topk_pprs[v][k];
            if (p.second > 0) {
                exact_ppr_map.insert(p);
            }
        }
        min_ssppr = exact_topk_pprs[v].back().second;
        //cout<<"exact dht:\t";
        int size_e = min(config.k, (unsigned int) exact_topk_dhts[v].size());
        for (int i = 0; i < size_e; i++) {
            pair<int, double> &p = exact_topk_dhts[v][i];
            //cout<<p.first<<":"<<p.second<<"\t";
            iDCG += (pow(2, p.second) - 1) / (log(i + 2) / log(2));
            if (p.second > 0) {
                exact_map.insert(p);
            }
        }
        //cout<<endl;
        //cout<<"test dht:\t";
        for (int j = 0; j < exact_map.size(); ++j) {
            int node = topk_dhts[j].first;
            //cout<<topk_dhts[j].second;
            double dht_tmp = 0;
            if (exact_map.find(node) != exact_map.end()) {
                dht_tmp = exact_map[node];
            } else if (exact_ppr_map.find(node) != exact_ppr_map.end() && ppr_self.find(node) != ppr_self.end()) {
                dht_tmp = exact_ppr_map.at(node) / ppr_self.at(node);
            } else if (exact_ppr_map.find(node) != exact_ppr_map.end() && ppr_self.find(node) == ppr_self.end()) {
                // recompute ppr(t,t)
                double ppr_self_new = pair_wise_ppr(node, node, graph);
                dht_tmp = min(exact_ppr_map.at(node) / ppr_self_new,exact_topk_dhts[v][size_e-1].second);

            } else if (exact_ppr_map.find(node) == exact_ppr_map.end() && ppr_self.find(node) != ppr_self.end()) {
                // recompute ppr(s,t)
                double ppr_source = min(pair_wise_ppr(v, node, graph), min_ssppr);
                dht_tmp = min(ppr_source / ppr_self.at(node),exact_topk_dhts[v][size_e-1].second);
            } else if (exact_ppr_map.find(node) == exact_ppr_map.end() && ppr_self.find(node) == ppr_self.end()) {
                // recompute ppr(s,t) and ppr(t,t)
                double ppr_source = min(pair_wise_ppr(v, node, graph), min_ssppr);
                double ppr_self_new = pair_wise_ppr(node, node, graph);
                dht_tmp = min(ppr_source / ppr_self_new,exact_topk_dhts[v][size_e-1].second);
            }
            DCG += (pow(2, dht_tmp) - 1) / (log(j + 2) / log(2));
        }
        //cout<<endl;
        double NDCG = DCG / iDCG;
        INFO(DCG,iDCG, NDCG);
        if (exact_map.size() <= 1)
            return;
        result.topk_NDCG += NDCG;
        return;
    }
}

inline bool cmp(double x, double y) {
    return x > y;
}

// obtain the top-k ppr values from ppr map
double kth_ppr() {
    Timer tm(SORT_MAP);

    static vector<double> temp_ppr;
    temp_ppr.clear();
    temp_ppr.resize(ppr.occur.m_num);
    int nodeid;
    for (int i; i < ppr.occur.m_num; i++) {
        // if(ppr.m_data[i]>0)
        // temp_ppr[size++] = ppr.m_data[i];
        temp_ppr[i] = ppr[ppr.occur[i]];
    }

    nth_element(temp_ppr.begin(), temp_ppr.begin() + config.k - 1, temp_ppr.end(), cmp);
    return temp_ppr[config.k - 1];

}

double topk_ppr() {
    topk_pprs.clear();
    topk_pprs.resize(config.k);

    static unordered_map<int, double> temp_ppr;
    temp_ppr.clear();
    // temp_ppr.resize(ppr.occur.m_num);
    int nodeid;
    for (long i = 0; i < ppr.occur.m_num; i++) {
        nodeid = ppr.occur[i];
        // INFO(nodeid);
        temp_ppr[nodeid] = ppr[nodeid];
    }

    partial_sort_copy(temp_ppr.begin(), temp_ppr.end(), topk_pprs.begin(), topk_pprs.end(),
                      [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

    return topk_pprs[config.k - 1].second;
}

double topk_dht() {
    topk_dhts.clear();
    topk_dhts.resize(config.k);

    static vector<pair<int, double> > temp_dht;
    temp_dht.clear();
    temp_dht.resize(dht.cur);
    int nodeid, cur = 0;
    for (int k = 0; k < dht.occur.m_num; ++k) {
        nodeid = dht.occur[k];
        if (dht.exist(nodeid))
            temp_dht[cur++] = MP(nodeid, dht[nodeid]);
    }

    partial_sort_copy(temp_dht.begin(), temp_dht.end(), topk_dhts.begin(), topk_dhts.end(),
                      [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

    return topk_dhts[config.k - 1].second;
}

void compute_precision_for_dif_k_dht(int v) {
    if (exact_topk_dhts.size() > 1 && exact_topk_dhts.find(v) != exact_topk_dhts.end()) {
        //vector<double> true_dht(exact_topk_dhts.size())
        for (auto k: ks) {

            int j = 0;
            unordered_map<int, double> topk_map;
            for (auto &p: topk_dhts) {
                if (p.second > 0) {
                    topk_map.insert(p);
                }
                j++;
                if (j == k) { // only pick topk
                    break;
                }
            }

            double recall = 0.0;
            unordered_map<int, double> exact_map;
            int size_e = min(k, (int) exact_topk_dhts[v].size());
            for (int i = 0; i < size_e; i++) {
                pair<int, double> &p = exact_topk_dhts[v][i];
                if (p.second > 0) {
                    exact_map.insert(p);
                    if (topk_map.find(p.first) != topk_map.end())
                        recall++;
                }
            }

            double precision = 0.0;
            for (auto &p: topk_map) {
                if (exact_map.find(p.first) != exact_map.end()) {
                    precision++;
                }
            }

            if (exact_map.size() <= 1)
                continue;

            precision = precision * 1.0 / exact_map.size();
            recall = recall * 1.0 / exact_map.size();

            pred_results[k].topk_precision += precision;
            pred_results[k].topk_recall += recall;
            pred_results[k].real_topk_source_count++;
        }
    }
}
void compute_NDCG_for_dif_k_dht(int v, const map<int, double> ppr_self, Graph &graph) {
    if (exact_topk_dhts.size() > 1 && exact_topk_dhts.find(v) != exact_topk_dhts.end()) {
        //vector<double> true_dht(exact_topk_dhts.size())
        unordered_map<int, double> exact_map, exact_ppr_map;
        for (int i1 = 0; i1 < exact_topk_pprs[v].size(); ++i1) {
            pair<int, double> &p = exact_topk_pprs[v][i1];
            if (p.second > 0) {
                exact_ppr_map.insert(p);
            }
        }
        for (auto k: ks) {
            double iDCG = 0, DCG = 0, min_ssppr = 0;
            //得到ppr的哈希表
            min_ssppr = exact_topk_pprs[v].back().second;
            // first comput IDCG
            int size_e = min((unsigned int)k, (unsigned int) exact_topk_dhts[v].size());
            for (int i = 0; i < size_e; i++) {
                pair<int, double> &p = exact_topk_dhts[v][i];
                iDCG += (pow(2, p.second) - 1) / (log(i + 2) / log(2));
                if (p.second > 0) {
                    exact_map.insert(p);
                }
            }
            int size_dht_real=min(k,int(exact_map.size()));
            for (int j = 0; j < size_dht_real; ++j) {
                int node = topk_dhts[j].first;
                double dht_tmp = 0;
                if (exact_map.find(node) != exact_map.end()) {
                    dht_tmp = exact_map[node];
                } else if (exact_ppr_map.find(node) != exact_ppr_map.end() && ppr_self.find(node) != ppr_self.end()) {
                    dht_tmp = exact_ppr_map.at(node) / ppr_self.at(node);
                } else if (exact_ppr_map.find(node) != exact_ppr_map.end() && ppr_self.find(node) == ppr_self.end()) {
                    // recompute ppr(t,t)
                    double ppr_self_new = pair_wise_ppr(node, node, graph);
                    dht_tmp = min(exact_ppr_map.at(node) / ppr_self_new,exact_topk_dhts[v][size_e-1].second);

                } else if (exact_ppr_map.find(node) == exact_ppr_map.end() && ppr_self.find(node) != ppr_self.end()) {
                    // recompute ppr(s,t)
                    double ppr_source = min(pair_wise_ppr(v, node, graph), min_ssppr);
                    dht_tmp = min(ppr_source / ppr_self.at(node),exact_topk_dhts[v][size_e-1].second);
                } else if (exact_ppr_map.find(node) == exact_ppr_map.end() && ppr_self.find(node) == ppr_self.end()) {
                    // recompute ppr(s,t) ppr(t,t)
                    double ppr_source = min(pair_wise_ppr(v, node, graph), min_ssppr);
                    double ppr_self_new = pair_wise_ppr(node, node, graph);
                    dht_tmp = min(ppr_source / ppr_self_new,exact_topk_dhts[v][size_e-1].second);
                }
                DCG += (pow(2, dht_tmp) - 1) / (log(j + 2) / log(2));
            }
            double NDCG = DCG / iDCG;
            INFO(k,NDCG);
            pred_results[k].topk_NDCG += NDCG;
        }
    }
}
inline void display_precision_for_dif_k() {
    split_line();
    cout << config.algo << endl;
    for (auto k: ks) {
        cout << k << "\t";
    }
    cout << endl << "Precision:" << endl;
    assert(pred_results[k].real_topk_source_count > 0);
    for (auto k: ks) {
        cout << pred_results[k].topk_precision / pred_results[k].real_topk_source_count << "\t";
    }
    cout << endl << "Recall:" << endl;
    for (auto k: ks) {
        cout << pred_results[k].topk_recall / pred_results[k].real_topk_source_count << "\t";
    }
    cout << endl << "NDCG:" << endl;
    for (auto k: ks) {
        cout << pred_results[k].topk_NDCG / pred_results[k].real_topk_source_count << "\t";
    }
    cout << endl;
}

int
reverse_local_update_linear_dht(int t, const Graph &graph, vector<int> &idx, vector<int> &node_with_r, int &pointer_r,
                                vector<int> &q, double init_residual = 1) {
    Timer tm(111);
    int backward_counter = 0, pointer_q = 0;
    //vector<int> q;
    //q.reserve(graph.n);
    //q.push_back(-1);
    unsigned long left = 0;

    double myeps = config2.rmax;
    q[pointer_q++] = t;
    //q.push_back(t);
    bwd_idx.second.occur[t] = t;
    bwd_idx.second[t] = 1;
    pointer_r = 0;
    node_with_r[pointer_r++] = t;
    idx[t] = t;
    while (left != pointer_q) {
        int v = q[left];
        idx[v] = -1;
        left++;
        left %= graph.n;
        if (bwd_idx.second[v] < myeps)
            break;

        if (v == t) {
            if (bwd_idx.first.occur[v] != t) {
                bwd_idx.first.occur[v] = t;
                bwd_idx.first[v] = bwd_idx.second[v] * config2.alpha;
            } else
                bwd_idx.first[v] += bwd_idx.second[v] * config.alpha;
        }

        double residual = (1 - config2.alpha) * bwd_idx.second[v];
        bwd_idx.second[v] = 0;
        if (graph.gr[v].size() > 0) {
            backward_counter += graph.gr[v].size();
            for (int next : graph.gr[v]) {
                int cnt = graph.g[next].size();
                if (bwd_idx.second.occur[next] != t) {
                    bwd_idx.second.occur[next] = t;
                    bwd_idx.second[next] = residual / cnt;
                    node_with_r[pointer_r++] = next;
                } else
                    bwd_idx.second[next] += residual / cnt;

                if (bwd_idx.second[next] > myeps && idx[next] != t) {
                    // put next into q if next is not in q
                    idx[next] = t;//(int) q.size();
                    //q.push_back(next);
                    q[pointer_q++] = next;
                    pointer_q %= graph.n;
                }
            }
        }
    }
    return backward_counter;
}

static void reverse_local_update_linear(int t, const Graph &graph, double init_residual = 1) {
    bwd_idx.first.clean();
    bwd_idx.second.clean();

    static unordered_map<int, bool> idx;
    idx.clear();

    vector<int> q;
    q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;

    double myeps = config.rmax;

    q.push_back(t);
    bwd_idx.second.insert(t, init_residual);

    idx[t] = true;
    while (left < q.size()) {
        int v = q[left];
        idx[v] = false;
        left++;
        if (bwd_idx.second[v] < myeps)
            break;

        if (bwd_idx.first.notexist(v))
            bwd_idx.first.insert(v, bwd_idx.second[v] * config.alpha);
        else
            bwd_idx.first[v] += bwd_idx.second[v] * config.alpha;

        double residual = (1 - config.alpha) * bwd_idx.second[v];
        bwd_idx.second[v] = 0;
        if (graph.gr[v].size() > 0) {
            for (int next : graph.gr[v]) {
                int cnt = graph.g[next].size();
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
        }
    }
}

int
reverse_local_update_linear_dht_topk(int s, const Graph &graph, double lowest_rmax, vector<int> &in_backward,
                                     vector<int> &in_next_backward, unordered_map<int, vector<int>> &backward_from) {
    Timer timer(BWD_LU);
    int backward_counter = 0;
    //init
    if (backward_from[s].empty()) {
        multi_bwd_idx_p[s] = 0;
        multi_bwd_idx_r[s][s] = 1.0;
        backward_from[s].push_back(s);
    }
    vector<int> next_backward_from;
    next_backward_from.reserve(graph.n);

    for (auto &v: backward_from[s]) {
        in_backward[v] = s;
    }
    unsigned long i = 0;
    while (i < backward_from[s].size()) {
        int v = backward_from[s][i++];
        in_backward[v] = s;
        if (multi_bwd_idx_r[s][v] >= config2.rmax) {
            int out_neighbor = graph.gr[v].size();
            backward_counter += out_neighbor;
            if (s == v) {
                assert(multi_bwd_idx_r[s][v]>0);
                multi_bwd_idx_p[s] += multi_bwd_idx_r[s][v] * config.alpha;
                if (multi_bwd_idx_p[s]==0){
                    INFO(multi_bwd_idx_r[s][v],s,v);
                }
            }
            double v_residue = (1 - config.alpha) * multi_bwd_idx_r[s][v];
            multi_bwd_idx_r[s].erase(v);
            if (out_neighbor > 0) {
                for (int nei: graph.gr[v]) {
                    int cnt = graph.g[nei].size();
                    multi_bwd_idx_r[s][nei] += v_residue / cnt;
                    if (in_backward[nei] != s && multi_bwd_idx_r[s][nei] >= config2.rmax) {
                        backward_from[s].push_back(nei);
                        in_backward[nei] = s;
                    } else if (in_next_backward[nei] != s && multi_bwd_idx_r[s][nei] >= lowest_rmax) {
                        next_backward_from.push_back(nei);
                        in_next_backward[nei] = s;
                    }
                }
            }
        } else if (in_next_backward[v] != s && multi_bwd_idx_r[s][v] >= lowest_rmax) {
            next_backward_from.push_back(v);
            in_next_backward[v] = s;
        }
    }
    backward_from[s] = next_backward_from;
    return backward_counter;
#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}


void init_bounds_self(const Graph &graph) {

    upper_bounds_self.reset_values(1.0);
    lower_bounds_self.reset_values(config2.alpha);
    /*
    upper_bounds_self.reset_values(1.0/(2-config2.alpha));
    lower_bounds_self.reset_values(config2.alpha);
    for (int k = 0; k < graph.n; ++k) {
        if (graph.g[k].empty()){
            upper_bounds_self[k]=1;
        }
    }
 */
}

void forward_local_update_linear(int s, const Graph &graph, double &rsum, double rmax, double init_residual = 1.0) {
    fwd_idx.first.clean();//p
    fwd_idx.second.clean();//r

    static vector<bool> idx(graph.n);//标志是否在队列中
    std::fill(idx.begin(), idx.end(), false);

    double myeps = rmax;//config.rmax;

    vector<int> q;  //nodes that can still propagate forward
    q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;
    q.push_back(s);

    // residual[s] = init_residual;
    fwd_idx.second.insert(s, init_residual);

    idx[s] = true;

    unsigned long long forward_counter = 0;
    while (left < (int) q.size()) {
        int v = q[left];
        idx[v] = false;
        left++;
        double v_residue = fwd_idx.second[v];
        fwd_idx.second[v] = 0;
        if (!fwd_idx.first.exist(v))//判断是否向量p中有点v，并进行更新
            fwd_idx.first.insert(v, v_residue * config.alpha);
        else
            fwd_idx.first[v] += v_residue * config.alpha;

        int out_neighbor = graph.g[v].size();
        rsum -= v_residue * config.alpha;
        if (out_neighbor == 0) {//如果邻居点没有出度，那么将这一点儿残差返回给源点！
            fwd_idx.second[s] += v_residue * (1 - config.alpha);
            if (graph.g[s].size() > 0 && fwd_idx.second[s] / graph.g[s].size() >= myeps && idx[s] != true) {
                idx[s] = true;
                q.push_back(s);
            }
            continue;
        }

        double avg_push_residual = ((1.0 - config.alpha) * v_residue) / out_neighbor;
        for (int next : graph.g[v]) {
            // total_push++;
            if (!fwd_idx.second.exist(next))
                fwd_idx.second.insert(next, avg_push_residual);
            else
                fwd_idx.second[next] += avg_push_residual;

            //if a node's' current residual is small, but next time it got a laerge residual, it can still be added into forward list
            //so this is correct
            if (fwd_idx.second[next] / graph.g[next].size() >= myeps && idx[next] != true) {
                idx[next] = true;//(int) q.size();
                q.push_back(next);
            }
        }
    }
}


int forward_local_update_linear_topk_dht2(int s, const Graph &graph, double &rsum, double rmax, double lowest_rmax,
                                          vector<int> &forward_from) {
    double myeps = rmax;

    static vector<bool> in_forward(graph.n);
    static vector<bool> in_next_forward(graph.n);

    std::fill(in_forward.begin(), in_forward.end(), false);
    std::fill(in_next_forward.begin(), in_next_forward.end(), false);

    vector<int> next_forward_from;
    next_forward_from.reserve(graph.n);
    for (auto &v: forward_from)
        in_forward[v] = true;
    unsigned long long forward_counter = 0;
    unsigned long i = 0;
    while (i < forward_from.size()) {
        int v = forward_from[i];
        i++;
        in_forward[v] = false;
        if (fwd_idx.second[v] / graph.g[v].size() >= myeps) {
            int out_neighbor = graph.g[v].size();
            forward_counter += out_neighbor;
            double v_residue = fwd_idx.second[v];
            fwd_idx.second[v] = 0;
            if (!fwd_idx.first.exist(v)) {
                fwd_idx.first.insert(v, v_residue * config.alpha);
            } else {
                fwd_idx.first[v] += v_residue * config.alpha;
            }

            rsum -= v_residue * config.alpha;
            if (out_neighbor == 0) {
                fwd_idx.second[s] += v_residue * (1 - config.alpha);
                if (graph.g[s].size() > 0 && in_forward[s] != true && fwd_idx.second[s] / graph.g[s].size() >= myeps) {
                    forward_from.push_back(s);
                    in_forward[s] = true;
                } else {
                    if (graph.g[s].size() >= 0 && in_next_forward[s] != true &&
                        fwd_idx.second[s] / graph.g[s].size() >= lowest_rmax) {
                        next_forward_from.push_back(s);
                        in_next_forward[s] = true;
                    }
                }
                continue;
            }
            double avg_push_residual = ((1 - config.alpha) * v_residue) / out_neighbor;
            for (int next: graph.g[v]) {
                if (!fwd_idx.second.exist(next))
                    fwd_idx.second.insert(next, avg_push_residual);
                else
                    fwd_idx.second[next] += avg_push_residual;

                if (in_forward[next] != true && fwd_idx.second[next] / graph.g[next].size() >= myeps) {
                    forward_from.push_back(next);
                    in_forward[next] = true;
                } else {
                    if (in_next_forward[next] != true && fwd_idx.second[next] / graph.g[next].size() >= lowest_rmax) {
                        next_forward_from.push_back(next);
                        in_next_forward[next] = true;
                    }
                }
            }
        } else {
            if (in_next_forward[v] != true && fwd_idx.second[v] / graph.g[v].size() >= lowest_rmax) {
                next_forward_from.push_back(v);
                in_next_forward[v] = true;
            }
        }
    }
    //INFO(forward_counter);
    num_total_fo += forward_counter;

    forward_from = next_forward_from;
    return forward_counter;
}


extern double threshold;

bool if_stop_dht(const unordered_map<int, bool> &candidate, double raw_epsilon,double min_delta) {
    //思考这两个条件
    //INFO(kth_ppr(),2.0 * config.delta / config.alpha);
    //if (kth_ppr() >= 2.0 * config.delta / config.alpha) return true;
    if (config.delta >= threshold) return false;

    double true_epsilon = raw_epsilon;//(1 + config.epsilon) / (1 - config2.epsilon) - 1;

    double error = 1.0 + true_epsilon;

    topk_dhts.clear();
    topk_dhts.resize(config.k);
    topk_filter.clean();

    static vector<pair<int, double> > temp_bounds;
    temp_bounds.clear();
    temp_bounds.resize(candidate.size());
    int nodeid, cur = 0;
    for (auto item:candidate) {
        nodeid = item.first;
        temp_bounds[cur++] = MP(nodeid, lower_bounds_dht[nodeid]);
    }
    //sort topk nodes by lower bound
    partial_sort_copy(temp_bounds.begin(), temp_bounds.end(), topk_dhts.begin(), topk_dhts.end(),
                      [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });
    //display_topk_dht();
    //for topk nodes, upper-bound/low-bound <= 1+epsilon

    double ratio = 0.0;
    double largest_ratio = 0.0;
    for (auto &node: topk_dhts) {
        topk_filter.insert(node.first, 1);
        ratio = upper_bounds_dht[node.first] / lower_bounds_dht[node.first];
        if (largest_ratio < ratio) {
            largest_ratio = ratio;
        }
        if (ratio > error) {
            /*
            INFO(ratio, error);
            cout<<upper_bounds_dht[node.first]<<"\t"<<lower_bounds_dht[node.first]<<"\t"<<
                 upper_bounds_self[node.first]<<"\t"<< lower_bounds_self[node.first]<<"\t"<<upper_bounds[node.first]<<"\t"<<
                 lower_bounds[node.first]<<"\t"<<node.first<<endl;*/
            return false;
        }
    }
    INFO(largest_ratio);
    // INFO("ratio checking passed--------------------------------------------------------------");
    //for remaining NO. k+1 to NO. n nodes, low-bound of k > the max upper-bound of remaining nodes
    /*int actual_exist_ppr_num = lower_bounds.occur.m_num;
    if(actual_exist_ppr_num == 0) return true;
    int actual_k = min(actual_exist_ppr_num-1, (int) config.k-1);
    double low_bound_k = topk_pprs[actual_k].second;*/
    double low_bound_k = topk_dhts[config.k - 1].second;
    if (low_bound_k <= config.delta) {
        return false;
    }
    for (auto item:candidate) {
        nodeid = item.first;
        if (topk_filter.exist(nodeid) || dht[nodeid] <= 0)
            continue;

        double upper_temp = upper_bounds_dht[nodeid];
        double lower_temp = lower_bounds_dht[nodeid];
        if (upper_temp > low_bound_k * error) {
            if (upper_temp > (1 + true_epsilon) / (1 - true_epsilon) * lower_temp)
                continue;
            else {
                return false;
            }
        } else {
            continue;
        }
    }

    return true;

}

inline double calculate_lambda(double rsum, double pfail, double upper_bound, long total_rw_num) {
    return 1.0 / 3 * log(2 / pfail) * rsum / total_rw_num +
           sqrt(4.0 / 9.0 * log(2.0 / pfail) * log(2.0 / pfail) * rsum * rsum +
                8 * total_rw_num * log(2.0 / pfail) * rsum * upper_bound)
           / 2.0 / total_rw_num;
}

double zero_ppr_upper_bound = 1.0;
double threshold = 0.0;

void set_ppr_bounds(const Graph &graph, double rsum, long total_rw_num) {
    Timer tm(100);

    const static double min_ppr = 1.0 / graph.n;
    const static double sqrt_min_ppr = sqrt(1.0 / graph.n);


    double epsilon_v_div = sqrt(2.67 * rsum * log(2.0 / config.pfail) / total_rw_num);
    double default_epsilon_v = epsilon_v_div / sqrt_min_ppr;

    int nodeid;
    double ub_eps_a;
    double lb_eps_a;
    double ub_eps_v;
    double lb_eps_v;
    double up_bound;
    double low_bound;
    // INFO(total_rw_num);
    // INFO(zero_ppr_upper_bound);
    //INFO(rsum, 1.0/config.pfail, log(2/config.pfail), zero_ppr_upper_bound, total_rw_num);
    zero_ppr_upper_bound = calculate_lambda(rsum, config.pfail, zero_ppr_upper_bound, total_rw_num);
    double large_ratio_f=0;
    for (long i = 0; i < ppr.occur.m_num; i++) {
        nodeid = ppr.occur[i];
        assert(ppr[nodeid] > 0);
        if (ppr[nodeid] <= 0)
            continue;
        double reserve = 0.0;
        if (fwd_idx.first.exist(nodeid))
            reserve = fwd_idx.first[nodeid];
        double epsilon_a = 1.0;
        if (upper_bounds.exist(nodeid)) {
            assert(upper_bounds[nodeid] > 0.0);
            if (upper_bounds[nodeid] > reserve)
                //epsilon_a = calculate_lambda( rsum, config.pfail, upper_bounds[nodeid] - reserve, total_rw_num);
                epsilon_a = calculate_lambda(rsum, config.pfail, upper_bounds[nodeid] - reserve, total_rw_num);
            else
                epsilon_a = calculate_lambda(rsum, config.pfail, 1 - reserve, total_rw_num);
        } else {
            /*if(zero_ppr_upper_bound > reserve)
                epsilon_a = calculate_lambda( rsum, config.pfail, zero_ppr_upper_bound-reserve, total_rw_num);
            else
                epsilon_a = calculate_lambda( rsum, config.pfail, 1.0-reserve, total_rw_num);*/
            epsilon_a = calculate_lambda(rsum, config.pfail, 1.0 - reserve, total_rw_num);
        }

        ub_eps_a = ppr[nodeid] + epsilon_a;
        lb_eps_a = ppr[nodeid] - epsilon_a;
        if (!(lb_eps_a > 0))
            lb_eps_a = 0;

        double epsilon_v = default_epsilon_v;
        if (fwd_idx.first.exist(nodeid) && fwd_idx.first[nodeid] > min_ppr) {
            if (lower_bounds.exist(nodeid))
                reserve = max(reserve, lower_bounds[nodeid]);
            epsilon_v = epsilon_v_div / sqrt(reserve);
        } else {
            if (lower_bounds[nodeid] > 0)
                epsilon_v = epsilon_v_div / sqrt(lower_bounds[nodeid]);
        }


        ub_eps_v = 1.0;
        lb_eps_v = 0.0;
        if (1.0 - epsilon_v > 0) {
            ub_eps_v = ppr[nodeid] / (1.0 - epsilon_v);
            lb_eps_v = ppr[nodeid] / (1.0 + epsilon_v);
        }

        up_bound = min(min(ub_eps_a, ub_eps_v), 1.0);
        low_bound = max(max(lb_eps_a, lb_eps_v), reserve);
        if (up_bound > 0) {
            if (!upper_bounds.exist(nodeid))
                upper_bounds.insert(nodeid, up_bound);
            else
                upper_bounds[nodeid] = up_bound;
        }

        if (low_bound >= 0) {
            if (!lower_bounds.exist(nodeid))
                lower_bounds.insert(nodeid, low_bound);
            else
                lower_bounds[nodeid] = low_bound;
        }
    }
}

void set_ppr_self_bounds(const Graph &graph, const unordered_map<int, bool> &candidate) {
    Timer tm(100);

    const static double min_ppr = 1.0 / graph.n;
    const static double sqrt_min_ppr = sqrt(1.0 / graph.n);


    double epsilon_v_div = sqrt(
            (2 + config2.epsilon * 2 / 3) * config2.rmax * log(2.0 / config2.pfail) / config2.omega);
    double default_epsilon_v = epsilon_v_div / sqrt_min_ppr;

    int nodeid;
    double ub_eps_a;
    double lb_eps_a;
    double ub_eps_v;
    double lb_eps_v;
    double up_bound;
    double low_bound;
    // INFO(total_rw_num);
    // INFO(zero_ppr_upper_bound);
    //INFO(rsum, 1.0/config.pfail, log(2/config.pfail), zero_ppr_upper_bound, total_rw_num);
    zero_ppr_upper_bound = calculate_lambda(config2.rmax, config2.pfail, zero_ppr_upper_bound, config2.omega);
    double large_ratio = 0;
    for (auto item:candidate) {
        int nodeid = item.first;
        assert(ppr_bi[nodeid] > 0);
        if (ppr_bi[nodeid] <= 0) continue;
        double reserve = 0;
        if (multi_bwd_idx_p[nodeid] > 0)
            reserve = multi_bwd_idx_p[nodeid];
        double epsilon_a = 1.0;
        if (upper_bounds_self.exist(nodeid)) {
            if (upper_bounds_self[nodeid] > reserve)
                //epsilon_a = calculate_lambda( rsum, config.pfail, upper_bounds[nodeid] - reserve, total_rw_num);
                epsilon_a = calculate_lambda(config2.rmax, config.pfail, upper_bounds_self[nodeid], config2.omega);
            else
                epsilon_a = calculate_lambda(config2.rmax, config.pfail, 1, config2.omega);
        } else {
            epsilon_a = calculate_lambda(config2.rmax, config.pfail, 1, config2.omega);
        }
        //INFO(epsilon_a,calculate_lambda(config2.rmax, config.pfail, upper_bounds_self[nodeid], config2.omega));
        ub_eps_a = ppr_bi[nodeid] + epsilon_a;
        lb_eps_a = ppr_bi[nodeid] - epsilon_a;
        if (!(lb_eps_a > 0))
            lb_eps_a = 0;
        double epsilon_v = default_epsilon_v;
        if (lower_bounds_self.exist(nodeid))
            reserve = max(reserve, lower_bounds_self[nodeid]);
        epsilon_v = epsilon_v_div / sqrt(reserve);
        //INFO(lower_bounds_self.exist(nodeid),lower_bounds_self[nodeid],epsilon_v);

        ub_eps_v = 1.0;
        lb_eps_v = 0.0;
        if (1.0 - epsilon_v > 0) {
            ub_eps_v = ppr_bi[nodeid] / (1.0 - epsilon_v);
            lb_eps_v = ppr_bi[nodeid] / (1.0 + epsilon_v);
        }

        up_bound = min(min(ub_eps_a, ub_eps_v), 1.0);
        low_bound = max(max(lb_eps_a, lb_eps_v), reserve);
        double old_up = upper_bounds_self[nodeid];
        double old_low = lower_bounds_self[nodeid];
        if (up_bound > 0) {
            if (!upper_bounds_self.exist(nodeid))
                upper_bounds_self.insert(nodeid, up_bound);
            else
                upper_bounds_self[nodeid] = up_bound;
        }

        if (low_bound >= 0) {
            if (!lower_bounds_self.exist(nodeid))
                lower_bounds_self.insert(nodeid, low_bound);
            else
                lower_bounds_self[nodeid] = low_bound;
        }

    }
}

void set_dht_bounds(unordered_map<int, bool> &candidate) {
    //先计算界限，然后找第k大的，然后更新
    //初始化
    static vector<double> temp_dht;
    temp_dht.clear();
    if (candidate.empty()) {
        temp_dht.resize(upper_bounds.occur.m_num);
        for (int j = 0; j < ppr.occur.m_num; ++j) {
            int node = ppr.occur[j];
            assert(ppr.exist(node));
            //dht.insert(node, ppr[node] / ppr_bi[node]);
            upper_bounds_dht.insert(node, upper_bounds[node] / lower_bounds_self[node]);
            lower_bounds_dht.insert(node, lower_bounds[node] / upper_bounds_self[node]);
            temp_dht[j] = lower_bounds_dht[node];
        }
        nth_element(temp_dht.begin(), temp_dht.begin() + config.k - 1, temp_dht.end(), cmp);
        double kth_dht_lb = temp_dht[config.k - 1];
        INFO(kth_dht_lb);
        for (int k = 0; k < ppr.occur.m_num; ++k) {
            int nodeid = ppr.occur[k];
            if (upper_bounds_dht[nodeid] >= kth_dht_lb) {
                candidate.insert(make_pair(nodeid, true));
            }
        }
    } else {
        temp_dht.resize(candidate.size());
        int cur = 0;
        for (auto item:candidate) {
            int node = item.first;
            assert(ppr.exist(node));
            //dht.insert(node, ppr[node] / ppr_bi[node]);
            assert(item.second);
            upper_bounds_dht.insert(node, upper_bounds[node] / lower_bounds_self[node]);
            lower_bounds_dht.insert(node, lower_bounds[node] / upper_bounds_self[node]);
            temp_dht[cur++] = lower_bounds_dht[node];
        }
        nth_element(temp_dht.begin(), temp_dht.begin() + config.k - 1, temp_dht.end(), cmp);
        double kth_dht_lb = temp_dht[config.k - 1];
        for (auto iter = candidate.begin(); iter != candidate.end();) {
            int node = iter->first;
            assert(upper_bounds_dht.exist(node));
            if (upper_bounds_dht[node] < kth_dht_lb) {
                iter = candidate.erase(iter);
                multi_bwd_idx_r.erase(node);
            } else {
                ++iter;
            }
        }
    }
}

#endif
