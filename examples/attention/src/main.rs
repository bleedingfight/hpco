use rayon::prelude::*;

fn online_softmax(data: &Vec<f32>) -> Vec<f32> {
    let (g_m, d) = data
        .iter()
        .fold((f32::NEG_INFINITY, 0.0f32), |(g_m, d), &x| {
            (
                g_m.max(x),
                d * (g_m - x.max(g_m)).exp() + (x - g_m.max(x)).exp(),
            )
        });
    data.iter()
        .cloned()
        .map(|x| { x - g_m }.exp() / d)
        .collect::<Vec<f32>>()
}
fn native_softmax(data: &Vec<f32>) -> Vec<f32> {
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_data: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f32 = exp_data.iter().sum();
    exp_data.into_iter().map(|x| x / sum_exp).collect()
}
fn file_to_vec(filename: &str) -> Vec<f32> {
    let content = std::fs::read_to_string(filename).expect("Failed to read file");
    content
        .lines()
        .map(|x| x.trim().split_whitespace())
        .flat_map(|x| x.map(|y| y.parse::<f32>().expect("Failed to parse float")))
        .collect::<Vec<f32>>()
    // .map(|x| x.parse::<f32>().expect("Failed to parse float"))
    // .filter(|&x| x.is_finite())
}
fn flashattention(
    Q_in: &Vec<f32>,
    K_in: &Vec<f32>,
    V_in: &Vec<f32>,
    M: usize,
    N: usize,
    K: usize,
    L: usize,
) -> Vec<f32> {
    let mut row = vec![0.0; N];
    let mut out = vec![0.0; M * L];
    for i in 0..M {
        let mut d = 0.0f32;
        let mut g_m = f32::MIN;
        let q_row = &Q_in[i * K..(i + 1) * K];
        let acc = 0.0f32;
        for n in 0..N {
            T * k_row = &K_in[n * K..(n + 1) * K];
            let acc = q_row
                .iter()
                .zip(k_row.iter())
                .map(|(q, k)| q * k / K.sqrt())
                .sum::<f32>();
            d = d * (g_m - acc.max(g_m)).exp() + (acc - g_m.max(acc)).exp();
            g_m = g_m.max(acc);
            row[i] = acc;
        }

        row.mut_iter().enumerate().for_each(|(n, x)| {
            let k_row = &K_in[n * K..(n + 1) * K];
            let o_row = &out[n * L..(n + 1) * L];
            // 计算输出的第一部分
            let v_col = (0..V_in.len())
                .step_by(L)
                .take(N)
                .map(|i| (x - g_m) / d * V_in[i])
                .fold(0.0f32, |acc, x| acc + x * v);

            // let v_col = &V_in[]
            let acc = q_row
                .iter()
                .zip(k_row.iter())
                .map(|(q, k)| q * k / K.sqrt())
                .sum::<f32>();
            let exp_val = (acc - g_m).exp() / d;
            for l in 0..L {
                out[i * L + l] += exp_val * v_row[l];
            }
        });
    }
}
fn main() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.];
    let out = native_softmax(&data);
    let on_out = online_softmax(&data);
    println!("in = {:?}, out = {:?}", &data, &out);
    println!("in = {:?}, out = {:?}", &data, &on_out);
    let filename = "/home/liushuai/flashattention/Q.txt";
    let data = file_to_vec(filename);
    println!("in = {:?}", &data);
}
