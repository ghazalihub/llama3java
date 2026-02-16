mod gguf;
mod config;
mod tensors;
mod tokenizer;
mod sampler;
mod transformer;

use clap::Parser;
use memmap2::MmapOptions;
use std::fs::File;
use std::io::{Write, BufRead};
use crate::gguf::GGUF;
use crate::config::Config;
use crate::transformer::{Weights, State, forward};
use crate::tokenizer::{Tokenizer, Vocabulary};
use crate::sampler::Sampler;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: String,

    #[arg(short, long)]
    prompt: Option<String>,

    #[arg(short, long)]
    chat: bool,

    #[arg(long, default_value_t = 0.1)]
    temp: f32,

    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    #[arg(short, long)]
    seed: Option<u64>,

    #[arg(long, default_value_t = true)]
    stream: bool,

    #[arg(long, default_value_t = false)]
    echo: bool,

    #[arg(short, long)]
    threads: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global()?;
    }

    println!("Loading model {}...", args.model);
    let gguf = GGUF::load(&args.model)?;
    let config = Config::from_gguf(&gguf);

    let file = File::open(&args.model)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    println!("Loading weights...");
    let weights = Weights::load(&config, &gguf, &mmap);

    println!("Initializing tokenizer...");
    let tokens = if let Some(gguf::MetadataValue::Array(arr)) = gguf.metadata.get("tokenizer.ggml.tokens") {
        arr.iter().map(|v| if let gguf::MetadataValue::String(s) = v { s.clone() } else { String::new() }).collect()
    } else {
        Vec::new()
    };
    let vocab = Vocabulary::new(tokens);

    let merges = if let Some(gguf::MetadataValue::Array(arr)) = gguf.metadata.get("tokenizer.ggml.merges") {
        arr.iter().map(|v| if let gguf::MetadataValue::String(s) = v { s.clone() } else { String::new() }).collect()
    } else {
        Vec::new()
    };

    let mut special_tokens = std::collections::HashMap::new();
    if let Some(id) = vocab.get_index("<|begin_of_text|>") { special_tokens.insert("<|begin_of_text|>".to_string(), id); }
    if let Some(id) = vocab.get_index("<|end_of_text|>") { special_tokens.insert("<|end_of_text|>".to_string(), id); }
    if let Some(id) = vocab.get_index("<|eot_id|>") { special_tokens.insert("<|eot_id|>".to_string(), id); }

    let tokenizer = Tokenizer::new(vocab, merges, special_tokens);
    let mut state = State::new(&config);
    let mut sampler = Sampler::new(args.seed);

    if args.chat {
        run_chat(&config, &mut state, &weights, &tokenizer, &mut sampler, args.temp, args.top_p, args.stream)?;
    } else if let Some(prompt) = args.prompt {
        run_instruct(&config, &mut state, &weights, &tokenizer, &mut sampler, &prompt, args.temp, args.top_p, args.stream, args.echo)?;
    } else {
        println!("Please provide a prompt or use --chat mode.");
    }

    Ok(())
}

fn run_instruct(config: &Config, state: &mut State, weights: &Weights, tokenizer: &Tokenizer, sampler: &mut Sampler, prompt: &str, temp: f32, top_p: f32, stream: bool, echo: bool) -> Result<(), Box<dyn std::error::Error>> {
    let prompt_tokens = tokenizer.encode(prompt, &[]);
    let mut pos = 0;
    let mut next = *tokenizer.special_tokens.get("<|begin_of_text|>").unwrap_or(&128000);

    let start_time = Instant::now();

    for &t in &prompt_tokens {
        forward(config, state, weights, next, pos);
        if echo {
            print!("{}", tokenizer.decode(&[t]));
            std::io::stdout().flush()?;
        }
        pos += 1;
        next = t;
    }

    while pos < config.seq_len {
        forward(config, state, weights, next, pos);
        next = sampler.sample(&mut state.logits, temp, top_p);

        if Some(&next) == tokenizer.special_tokens.get("<|end_of_text|>") { break; }
        if Some(&next) == tokenizer.special_tokens.get("<|eot_id|>") { break; }

        if stream {
            print!("{}", tokenizer.decode(&[next]));
            std::io::stdout().flush()?;
        }
        pos += 1;
    }

    let duration = start_time.elapsed();
    println!("\n\n{} tokens in {:.2}s ({:.2} tok/s)", pos, duration.as_secs_f32(), pos as f32 / duration.as_secs_f32());
    Ok(())
}

fn run_chat(config: &Config, state: &mut State, weights: &Weights, tokenizer: &Tokenizer, sampler: &mut Sampler, temp: f32, top_p: f32, stream: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut pos = 0;
    let mut next = *tokenizer.special_tokens.get("<|begin_of_text|>").unwrap_or(&128000);

    let stdin = std.io::stdin();
    let mut handle = stdin.lock();

    loop {
        print!("\n> ");
        std.io::stdout().flush()?;
        let mut line = String::new();
        handle.read_line(&mut line)?;
        let line = line.trim();
        if line.is_empty() || line == "quit" || line == "exit" { break; }

        let user_prompt = format!("<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", line);
        let prompt_tokens = tokenizer.encode(&user_prompt, &["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]);

        for &t in &prompt_tokens {
            forward(config, state, weights, next, pos);
            pos += 1;
            next = t;
        }

        while pos < config.seq_len {
            forward(config, state, weights, next, pos);
            next = sampler.sample(&mut state.logits, temp, top_p);

            if Some(&next) == tokenizer.special_tokens.get("<|eot_id|>") { break; }
            if Some(&next) == tokenizer.special_tokens.get("<|end_of_text|>") { break; }

            if stream {
                print!("{}", tokenizer.decode(&[next]));
                std.io::stdout().flush()?;
            }
            pos += 1;
        }
        println!();
    }
    Ok(())
}
