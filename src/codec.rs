use thiserror::Error;

const HEADER_LEN: usize = 14;
const END: [u8; 8] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01];
#[derive(Error, Debug)]
pub enum Error {
    #[error("The QOI decoder does not support decoding the format with magic bytes {0}")]
    InvalidMagicBytes(String),

    #[error("The image header is invalid")]
    InvalidHeader,

    #[error("The image end is invalid")]
    InvalidEnd,

    #[error("Could not read byte")]
    InvalidByte(#[from] std::io::Error),

    #[error(transparent)]
    Utf8(#[from] std::str::Utf8Error),

    #[error(transparent)]
    SliceParse(#[from] std::array::TryFromSliceError),
}

#[derive(Clone, Copy, Debug)]
pub struct Pixel([u8; 4]);

impl Pixel {
    fn new(r: u8, g: u8, b: u8, a: u8) -> Pixel {
        Pixel([r, g, b, a])
    }

    fn r(&self) -> u8 {
        self.0[0]
    }

    fn g(&self) -> u8 {
        self.0[1]
    }

    fn b(&self) -> u8 {
        self.0[2]
    }

    fn a(&self) -> u8 {
        self.0[3]
    }
}

impl PartialEq for Pixel {
    fn eq(&self, other: &Self) -> bool {
        self.r() == other.r()
            && self.g() == other.g()
            && self.b() == other.b()
            && self.a() == other.a()
    }
}

fn hash(pixel: &Pixel) -> usize {
    ((pixel.r() as u16 * 3 + pixel.g() as u16 * 5 + pixel.b() as u16 * 7 + pixel.a() as u16 * 11)
        % 64) as usize
}

struct Header {
    magic: [u8; 4],
    width: u32,
    height: u32,
    channels: u8,
    colorspace: u8,
}

impl Header {
    fn new(magic: [u8; 4], width: u32, height: u32, channels: u8, colorspace: u8) -> Self {
        Self {
            magic,
            width,
            height,
            channels,
            colorspace,
        }
    }

    fn to_bytes(self) -> Vec<u8> {
        self.magic
            .to_vec()
            .into_iter()
            .chain(self.width.to_be_bytes())
            .chain(self.height.to_be_bytes())
            .chain(self.channels.to_be_bytes())
            .chain(self.colorspace.to_be_bytes())
            .collect()
    }
}

impl TryFrom<&[u8; 14]> for Header {
    type Error = Error;

    fn try_from(bytes: &[u8; 14]) -> Result<Self, Self::Error> {
        let magic = std::str::from_utf8(&bytes[0..4])?;
        if magic != "qoif" {
            return Err(Error::InvalidMagicBytes(magic.to_string()));
        }
        let width = u32::from_be_bytes(bytes[4..8].try_into()?);
        let height = u32::from_be_bytes(bytes[8..12].try_into()?);
        let channels = u8::from_be_bytes(bytes[12..13].try_into()?);
        let colorspace = u8::from_be_bytes(bytes[13..14].try_into()?);

        Ok(Self {
            magic: bytes[0..4].try_into()?,
            width,
            height,
            channels,
            colorspace,
        })
    }
}

#[derive(Debug)]
enum OpCode {
    Run(u8),
    Index(u8),
    RGBA(u8, u8, u8, u8),
    Diff(u8, u8, u8),
    Luma(u8, u8, u8),
    RGB(u8, u8, u8),
}

impl OpCode {
    fn from_bytes<'a, I>(mut bytes: I) -> Option<Self>
    where
        I: Iterator<Item = &'a u8>,
    {
        use OpCode::*;

        let first_byte = bytes.next()?;
        match first_byte {
            x if x & 0b11111111 == 0b11111110 => {
                let r = bytes.next()?.clone();
                let g = bytes.next()?.clone();
                let b = bytes.next()?.clone();
                Some(RGB(r, g, b))
            }
            x if x & 0b11111111 == 0b11111111 => {
                let r = bytes.next()?.clone();
                let g = bytes.next()?.clone();
                let b = bytes.next()?.clone();
                let a = bytes.next()?.clone();
                Some(RGBA(r, g, b, a))
            }
            x if x & 0b11000000 == 0b00000000 => Some(Index((x & 0b00111111) as u8)),
            x if x & 0b11000000 == 0b01000000 => {
                // The encoder encodes unsigned integers only, so we remove the
                // artificially added bias of 2
                Some(Diff(
                    (x & 0b00110000) >> 4,
                    (x & 0b00001100) >> 2,
                    x & 0b00000011,
                ))
            }
            x if x & 0b11000000 == 0b10000000 => {
                let second_byte = bytes.next()?;

                let dg = x & 0b00111111;
                let dr_dg = (second_byte & 0b11110000) >> 4;
                let db_dg = second_byte & 0b00001111;
                Some(Luma(dg, dr_dg, db_dg))
            }
            x if x & 0b11000000 == 0b11000000 => Some(Run(x & 0b00111111)),
            _ => unreachable!(),
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        use OpCode::*;
        match self {
            RGBA(r, g, b, a) => {
                vec![0b11111111, r, g, b, a]
            }
            RGB(r, g, b) => {
                vec![0b11111110, r, g, b]
            }
            Run(run) => {
                assert!(run < 62);
                let biased_run = run & 0b00111111;
                vec![biased_run ^ 0b11000000]
            }
            Index(index) => {
                assert!(index < 64);
                vec![index & 0b00111111]
            }
            Diff(dr, dg, db) => {
                assert!(dr < 4);
                assert!(dg < 4);
                assert!(db < 4);
                vec![0b01000000 ^ (dr << 4) ^ (dg << 2) ^ db]
            }
            Luma(dg, dr_dg, db_dg) => {
                assert!(dg < 64);
                assert!(dr_dg < 16);
                assert!(db_dg < 16);
                vec![dg ^ 0b10000000, (dr_dg << 4) ^ db_dg]
            }
        }
    }

    fn len(&self) -> usize {
        use OpCode::*;
        match self {
            RGBA(_, _, _, _) => 5,
            RGB(_, _, _) => 4,
            Run(_) => 1,
            Index(_) => 1,
            Diff(_, _, _) => 1,
            Luma(_, _, _) => 2,
        }
    }
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

    pub fn decode(bytes: Vec<u8>) -> Result<Vec<Pixel>, Error> {
        if bytes.len() < HEADER_LEN + END.len() {
            return Err(Error::InvalidHeader);
        }

        let _ = Header::try_from(&bytes[0..HEADER_LEN].try_into()?)?;
        let mut decoder = Decoder::new();
        Ok(decoder.push(&bytes[HEADER_LEN..(bytes.len() - END.len())]))
    }

    pub fn push(&mut self, bytes: &[u8]) -> Vec<Pixel> {
        let mut index = 0;
        let mut pixels = vec![];
        while let Some(op_code) =
            OpCode::from_bytes(self.buffered_bytes.iter().chain(bytes).skip(index))
        {
            use OpCode::*;
            match op_code {
                RGBA(r, g, b, a) => pixels.push(Pixel::new(r, g, b, a)),
                RGB(r, g, b) => pixels.push(Pixel::new(r, g, b, self.previous_pixel.a())),
                Index(index) => pixels.push(self.seen_pixel_values[index as usize].clone()),
                Diff(dr, dg, db) => pixels.push(self.diff(dr, dg, db)),
                Luma(dg, dr_dg, db_dg) => pixels.push(self.luma(dg, dr_dg, db_dg)),
                Run(amount) => pixels
                    .extend(std::iter::repeat(self.previous_pixel).take((amount + 1) as usize)),
            }
            if let Some(last_pixel) = pixels.last() {
                self.previous_pixel = last_pixel.clone();
                self.seen_pixel_values[hash(&last_pixel)] = last_pixel.clone()
            }
            index += op_code.len();
        }

        self.buffered_bytes = bytes[(index - self.buffered_bytes.len())..].to_vec();
        pixels
    }

    fn diff(&self, dr: u8, dg: u8, db: u8) -> Pixel {
        Pixel::new(
            self.previous_pixel.r().wrapping_add(dr).wrapping_sub(2),
            self.previous_pixel.g().wrapping_add(dg).wrapping_sub(2),
            self.previous_pixel.b().wrapping_add(db).wrapping_sub(2),
            self.previous_pixel.a(),
        )
    }

    fn luma(&self, dg: u8, dr_dg: u8, db_dg: u8) -> Pixel {
        let g = self.previous_pixel.g().wrapping_add(dg).wrapping_sub(32);
        Pixel::new(
            self.previous_pixel
                .r()
                .wrapping_add(dr_dg)
                .wrapping_add(dg)
                .wrapping_sub(8 + 32),
            g,
            self.previous_pixel
                .b()
                .wrapping_add(db_dg)
                .wrapping_add(dg)
                .wrapping_sub(8 + 32),
            self.previous_pixel.a(),
        )
    }
}

struct Encoder {
    seen_pixel_values: [Pixel; 64],
    previous_pixel: Pixel,
    run_length: u8,
    pixel_index: u32,
}

enum Chainer {
    Continue(Vec<u8>),
    Stop(Vec<u8>),
}

impl Chainer {
    pub fn if_continue(self, chainer: Chainer) -> Self {
        use Chainer::*;
        match (self, chainer) {
            (Continue(mut bytes), Continue(mut chained_bytes)) => {
                bytes.append(&mut chained_bytes);
                Continue(bytes)
            }
            (Continue(mut bytes), Stop(mut chained_bytes)) => {
                bytes.append(&mut chained_bytes);
                Stop(bytes)
            }
            (Stop(bytes), _) => Stop(bytes),
        }
    }

    pub fn take(self) -> Vec<u8> {
        use Chainer::*;
        match self {
            Continue(bytes) | Stop(bytes) => bytes,
        }
    }
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

    pub fn encode(
        pixels: Vec<Pixel>,
        width: u32,
        height: u32,
        channels: u8,
        colorspace: u8,
    ) -> Vec<u8> {
        let mut encoder = Encoder::new();
        let header = Header::new(
            [b'q', b'o', b'i', b'f'],
            width,
            height,
            channels,
            colorspace,
        );

        let encoded_data = encoder.push(pixels, width * height);
        header
            .to_bytes()
            .into_iter()
            .chain(encoded_data.into_iter())
            .chain(END.into_iter())
            .collect()
    }

    pub fn push(&mut self, pixels: Vec<Pixel>, image_size: u32) -> Vec<u8> {
        pixels
            .into_iter()
            .flat_map(|pixel| {
                let mut bytes = self
                    .run(&pixel)
                    .if_continue(self.index(&pixel))
                    .if_continue(self.rgba(&pixel))
                    .if_continue(self.diff(&pixel))
                    .if_continue(self.luma(&pixel))
                    .if_continue(self.rgb(&pixel))
                    .take();
                self.pixel_index += 1;

                if self.pixel_index == image_size && self.run_length > 0 {
                    bytes.append(&mut OpCode::Run(self.run_length - 1).into_bytes());
                }

                self.seen_pixel_values[hash(&pixel)] = pixel.clone();
                self.previous_pixel = pixel;
                bytes
            })
            .collect()
    }

    fn rgba(&self, pixel: &Pixel) -> Chainer {
        use Chainer::*;
        if self.previous_pixel.a() != pixel.a() {
            Stop(OpCode::RGBA(pixel.r(), pixel.g(), pixel.b(), pixel.a()).into_bytes())
        } else {
            Continue(vec![])
        }
    }

    fn run(&mut self, pixel: &Pixel) -> Chainer {
        // The run opcode has a bias of -1
        use Chainer::*;
        if &self.previous_pixel == pixel {
            self.run_length += 1;
            if self.run_length > 62 {
                self.run_length = 1;
                return Stop(OpCode::Run(62 - 1).into_bytes());
            }
            return Stop(vec![]);
        } else if self.run_length > 0 {
            let run_length = self.run_length;
            self.run_length = 0;
            return Continue(OpCode::Run(run_length - 1).into_bytes());
        }

        Continue(vec![])
    }

    fn index(&self, pixel: &Pixel) -> Chainer {
        use Chainer::*;
        let index = hash(pixel);
        if let Some(seen_pixel) = self.seen_pixel_values.get(index) {
            if seen_pixel == pixel {
                return Stop(OpCode::Index(index as u8).into_bytes());
            }
        }
        Continue(vec![])
    }

    fn diff(&self, pixel: &Pixel) -> Chainer {
        use Chainer::*;
        let dr = pixel
            .r()
            .wrapping_sub(self.previous_pixel.r())
            .wrapping_add(2);
        let dg = pixel
            .g()
            .wrapping_sub(self.previous_pixel.g())
            .wrapping_add(2);
        let db = pixel
            .b()
            .wrapping_sub(self.previous_pixel.b())
            .wrapping_add(2);
        if dr < 4 && dg < 4 && db < 4 {
            Stop(OpCode::Diff(dr, dg, db).into_bytes())
        } else {
            Continue(vec![])
        }
    }

    fn luma(&self, pixel: &Pixel) -> Chainer {
        use Chainer::*;
        let dg = pixel
            .g()
            .wrapping_sub(self.previous_pixel.g())
            .wrapping_add(32);
        let dr_dg = pixel
            .r()
            .wrapping_sub(self.previous_pixel.r())
            .wrapping_sub(dg)
            .wrapping_add(8 + 32);
        let db_dg = pixel
            .b()
            .wrapping_sub(self.previous_pixel.b())
            .wrapping_sub(dg)
            .wrapping_add(8 + 32);
        if dg < 64 && dr_dg < 16 && db_dg < 16 {
            Stop(OpCode::Luma(dg, dr_dg, db_dg).into_bytes())
        } else {
            Continue(vec![])
        }
    }

    fn rgb(&mut self, pixel: &Pixel) -> Chainer {
        Chainer::Stop(OpCode::RGB(pixel.r(), pixel.g(), pixel.b()).into_bytes())
    }
}

#[cfg(test)]
mod tests {
    use image::io::Reader as ImageReader;
    use image::Pixel as PixelTrait;

    #[test]
    fn test_hash() {
        let pixel1 = super::Pixel([10, 20, 30, 255]);
        let pixel2 = super::Pixel([41, 210, 234, 255]);

        let bytes = super::Encoder::new().push(vec![pixel1, pixel2, pixel1], 3);
        assert!(matches!(
            super::OpCode::from_bytes(bytes.iter()),
            Some(super::OpCode::RGB(10, 20, 30)),
        ));
        assert!(matches!(
            super::OpCode::from_bytes(bytes.iter().skip(4)),
            Some(super::OpCode::RGB(41, 210, 234)),
        ));
        assert!(matches!(
            super::OpCode::from_bytes(bytes.iter().skip(8)),
            Some(super::OpCode::Index(9)),
        ));
        let decoded_pixels = super::Decoder::new().push(&bytes);
        assert_eq!(pixel1, decoded_pixels[0]);
        assert_eq!(pixel2, decoded_pixels[1]);
        assert_eq!(pixel1, decoded_pixels[2]);
    }

    #[test]
    fn test_diff() {
        let pixel1 = super::Pixel([10, 20, 30, 255]);
        let pixel2 = super::Pixel([11, 18, 31, 255]);

        let bytes = super::Encoder::new().push(vec![pixel1, pixel2], 2);
        assert!(matches!(
            super::OpCode::from_bytes(bytes.iter()),
            Some(super::OpCode::RGB(10, 20, 30)),
        ));
        assert!(matches!(
            super::OpCode::from_bytes(bytes.iter().skip(4)),
            Some(super::OpCode::Diff(3, 0, 3))
        ));
        let decoded_pixels = super::Decoder::new().push(&bytes);
        assert_eq!(pixel1, decoded_pixels[0]);
        assert_eq!(pixel2, decoded_pixels[1]);
    }

    #[test]
    fn test_luma() {
        let pixel1 = super::Pixel([10, 20, 30, 255]);
        let pixel2 = super::Pixel([42, 50, 57, 255]);

        let bytes = super::Encoder::new().push(vec![pixel1, pixel2], 2);
        assert!(matches!(
            super::OpCode::from_bytes(bytes.iter()),
            Some(super::OpCode::RGB(10, 20, 30)),
        ));
        assert!(matches!(
            super::OpCode::from_bytes(bytes.iter().skip(4)),
            Some(super::OpCode::Luma(62, 10, 5))
        ));
        let decoded_pixels = super::Decoder::new().push(&bytes);
        assert_eq!(pixel1, decoded_pixels[0]);
        assert_eq!(pixel2, decoded_pixels[1]);
    }

    #[test]
    fn test_auto_encode() {
        let png_path = "/Users/brinck10/Downloads/qoi_test_images/testcard.png";
        let png_image = ImageReader::open(png_path)
            .expect(&format!("Could not load image {}", png_path))
            .decode()
            .expect("Could not decode")
            .into_rgba8();
        let (width, height) = png_image.dimensions();

        let png_image = png_image
            .pixels()
            .map(|pixel| {
                let pixel = super::Pixel(pixel.channels().clone().try_into().unwrap());
                pixel
            })
            .collect::<Vec<_>>();
        let encoded_bytes = super::Encoder::encode(png_image.clone(), width, height, 4, 0);
        let auto_encoded_png_image = super::Decoder::decode(encoded_bytes).unwrap();
        assert_eq!(png_image, auto_encoded_png_image);
    }

    #[test]
    fn test_qoi_decode() {
        let png_path = "/Users/brinck10/Downloads/qoi_test_images/testcard.png";
        let png_image = ImageReader::open(png_path)
            .expect(&format!("Could not load image {}", png_path))
            .decode()
            .expect("Could not decode")
            .into_rgba8()
            .pixels()
            .map(|pixel| {
                let pixel = super::Pixel(pixel.channels().clone().try_into().unwrap());
                pixel
            })
            .collect::<Vec<_>>();

        let qoi_path = "/Users/brinck10/Downloads/qoi_test_images/testcard.qoi";
        let qoi_image = std::fs::read(qoi_path).unwrap();
        let decoded_qoi = super::Decoder::decode(qoi_image).unwrap();
        assert_eq!(png_image, decoded_qoi);
    }
}
