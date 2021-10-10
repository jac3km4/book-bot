use std::borrow::Cow;
use std::ffi::OsStr;
use std::fs::read_dir;
use std::path::Path;

use anyhow::Result;
use ordered_float::OrderedFloat;
use rust_bert::pipelines::pos_tagging::POSModel;
use rust_bert::pipelines::question_answering::{Answer, QaInput, QuestionAnsweringModel};

use crate::book::BookIndex;
use crate::util;

pub struct ChatBot {
    qa_model: QuestionAnsweringModel,
    pos_model: POSModel,
    books: Vec<BookIndex>,
}

impl ChatBot {
    const MAX_CONTEXT_PARAGRAPHS: usize = 32;
    const BATCH_SIZE: usize = 32;
    const ANSWER_COUNT: u32 = 8;

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let qa_model = QuestionAnsweringModel::new(Default::default())?;
        let pos_model = POSModel::new(Default::default())?;
        let mut books = vec![];

        for entry in read_dir(path)? {
            let entry = entry?;

            if entry.path().is_file() && entry.path().extension() == Some(OsStr::new("json")) {
                books.push(BookIndex::load(entry.path())?);
            }
        }

        let res = Self {
            qa_model,
            pos_model,
            books,
        };
        Ok(res)
    }

    pub fn answer(&self, question: String) -> Option<Answer> {
        let tags = util::extract_pos_tags(&question, &self.pos_model);

        let context = self
            .books
            .iter()
            .flat_map(|book| book.get_by_tags(&tags).take(Self::MAX_CONTEXT_PARAGRAPHS))
            .map(Cow::Borrowed)
            .reduce(|a, b| Cow::Owned(a.into_owned() + " " + b.as_ref()))?
            .into_owned();
        let input = QaInput { question, context };

        self.qa_model
            .predict(&[input], Self::ANSWER_COUNT.into(), Self::BATCH_SIZE)
            .first()?
            .iter()
            .filter(|answer| answer.score > 0.1)
            .max_by_key(|answer| OrderedFloat(Self::weight_function(answer)))
            .cloned()
    }

    fn weight_function(answer: &Answer) -> f64 {
        answer.answer.len() as f64 / 100.0 + answer.score
    }
}
