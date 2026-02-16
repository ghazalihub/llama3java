use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4,
    Q4_3 = 5,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
}

impl From<u32> for GGMLType {
    fn from(val: u32) -> Self {
        match val {
            0 => GGMLType::F32,
            1 => GGMLType::F16,
            2 => GGMLType::Q4_0,
            3 => GGMLType::Q4_1,
            8 => GGMLType::Q8_0,
            _ => panic!("Unsupported GGML type: {}", val),
        }
    }
}

impl GGMLType {
    pub fn type_size(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 => 18,
            GGMLType::Q8_0 => 34,
            _ => panic!("Unsupported GGML type size"),
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            GGMLType::F32 | GGMLType::F16 => 1,
            GGMLType::Q4_0 | GGMLType::Q8_0 => 32,
            _ => panic!("Unsupported GGML block size"),
        }
    }

    pub fn byte_size_for(&self, num_elements: usize) -> usize {
        let bsize = self.block_size();
        let tsize = self.type_size();
        assert_eq!(num_elements % bsize, 0);
        (num_elements / bsize) * tsize
    }
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8), Int8(i8), Uint16(u16), Int16(i16), Uint32(u32), Int32(i32),
    Float32(f32), Bool(bool), String(String), Array(Vec<MetadataValue>),
    Uint64(u64), Int64(i64), Float64(f64),
}

#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GGMLType,
    pub offset: u64,
}

impl GGUFTensorInfo {
    pub fn num_elements(&self) -> usize {
        self.dimensions.iter().product::<u64>() as usize
    }
}

pub struct GGUF {
    pub version: u32,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensor_infos: HashMap<String, GGUFTensorInfo>,
    pub tensor_data_offset: u64,
}

impl GGUF {
    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let magic = file.read_u32::<LittleEndian>()?;
        if magic != 0x46554747 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid GGUF magic"));
        }

        let version = file.read_u32::<LittleEndian>()?;
        let tensor_count = file.read_u64::<LittleEndian>()?;
        let metadata_count = file.read_u64::<LittleEndian>()?;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let key = read_string(&mut file)?;
            let val_type = file.read_u32::<LittleEndian>()?;
            let value = read_metadata_value(&mut file, val_type)?;
            metadata.insert(key, value);
        }

        let mut tensor_infos = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut file)?;
            let n_dims = file.read_u32::<LittleEndian>()?;
            let mut dims = Vec::new();
            for _ in 0..n_dims {
                dims.push(file.read_u64::<LittleEndian>()?);
            }
            let ggml_type = GGMLType::from(file.read_u32::<LittleEndian>()?);
            let offset = file.read_u64::<LittleEndian>()?;
            tensor_infos.insert(name.clone(), GGUFTensorInfo { name, dimensions: dims, ggml_type, offset });
        }

        let current_pos = file.stream_position()?;
        let alignment = if let Some(MetadataValue::Uint32(a)) = metadata.get("general.alignment") {
            *a as u64
        } else {
            32
        };
        let padding = (alignment - (current_pos % alignment)) % alignment;
        let tensor_data_offset = current_pos + padding;

        Ok(GGUF { version, metadata, tensor_infos, tensor_data_offset })
    }
}

fn read_string(file: &mut File) -> std::io::Result<String> {
    let len = file.read_u64::<LittleEndian>()?;
    let mut buf = vec![0; len as usize];
    file.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_metadata_value(file: &mut File, val_type: u32) -> std::io::Result<MetadataValue> {
    match val_type {
        0 => Ok(MetadataValue::Uint8(file.read_u8()?)),
        1 => Ok(MetadataValue::Int8(file.read_i8()?)),
        2 => Ok(MetadataValue::Uint16(file.read_u16::<LittleEndian>()?)),
        3 => Ok(MetadataValue::Int16(file.read_i16::<LittleEndian>()?)),
        4 => Ok(MetadataValue::Uint32(file.read_u32::<LittleEndian>()?)),
        5 => Ok(MetadataValue::Int32(file.read_i32::<LittleEndian>()?)),
        6 => Ok(MetadataValue::Float32(file.read_f32::<LittleEndian>()?)),
        7 => Ok(MetadataValue::Bool(file.read_u8()? != 0)),
        8 => Ok(MetadataValue::String(read_string(file)?)),
        9 => {
            let item_type = file.read_u32::<LittleEndian>()?;
            let len = file.read_u64::<LittleEndian>()?;
            let mut array = Vec::new();
            for _ in 0..len {
                array.push(read_metadata_value(file, item_type)?);
            }
            Ok(MetadataValue::Array(array))
        },
        10 => Ok(MetadataValue::Uint64(file.read_u64::<LittleEndian>()?)),
        11 => Ok(MetadataValue::Int64(file.read_i64::<LittleEndian>()?)),
        12 => Ok(MetadataValue::Float64(file.read_f64::<LittleEndian>()?)),
        _ => panic!("Unsupported metadata value type: {}", val_type),
    }
}
