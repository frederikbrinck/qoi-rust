struct Encoder {
    seen_pixel_values: [Pixel; 64],
    previous_pixel: Pixel,
    run_length: u8,
    pixel_index: u32,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            seen_pixel_values: [Pixel([0, 0, 0, 255]); 64],
            previous_pixel: Pixel([0, 0, 0, 255]),
            run_length: 0,
            pixel_index: 0,
        }
    }

    pub fn push(&mut self, pixels: Vec<Pixel>, image_size: u32) -> Vec<u8> {}
}

pub struct Decoder {
    previous_pixel: Pixel,
    seen_pixel_values: [Pixel; 64],
    buffered_bytes: Vec<u8>,
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            previous_pixel: Pixel([0, 0, 0, 255]),
            seen_pixel_values: [Pixel([0, 0, 0, 255]); 64],
            buffered_bytes: vec![],
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Vec<Pixel> {}
}
