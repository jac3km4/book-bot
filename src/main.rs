#![feature(slice_group_by)]

use std::fs::File;

use anyhow::Result;
use epubs::Epub;

use crate::book::BookIndex;
use crate::bot::ChatBot;

mod book;
mod bot;
mod util;

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();

    match &args[..] {
        [cmd, input, output] if cmd == "generate" => {
            let mut book = Epub::new(File::open(input)?)?;
            BookIndex::generate(&mut book)?.save(output)?;
        }
        [cmd, base_path, question] if cmd == "answer" => {
            let bot = ChatBot::load(base_path)?;

            println!("{}", question);
            match bot.answer(question.to_owned()) {
                Some(answer) => println!("{}", answer.answer),
                None => println!("I don't know"),
            }
        }
        _ => println!("Invalid arguments"),
    }
    Ok(())
}
