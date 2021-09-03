use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use darknet::{Image, Network};

const IMAGE_PATH: &'static str = "input/scientists.jpg";
const CFG_PATH: &'static str = "input/yolov4.cfg";
const WEIGHTS_PATH: &'static str = "input/yolov4.weights";

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut net = Network::load(CFG_PATH, Some(WEIGHTS_PATH), false).unwrap();
    let image = Image::open(IMAGE_PATH).unwrap();

    let mut group = c.benchmark_group("inference");
    group.sample_size(10);
    group.throughput(Throughput::Elements(1));
    group.bench_function("scientists", |b| {
        b.iter(|| {
            let detections = net.predict(&image, 0.25, 0.5, 0.45, true);
            assert_eq!(detections.len(), 164);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
