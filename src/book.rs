use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io;
use std::path::Path;

use anyhow::Result;
use epubs::{self, roxmltree, Epub, Href, NavPoint};
use phf::phf_set;
use rust_bert::pipelines::pos_tagging::POSModel;
use serde::{Deserialize, Serialize};

use crate::util;

#[derive(Debug, Serialize, Deserialize)]
pub struct BookIndex {
    paragraphs: Vec<String>,
    tags: HashMap<String, Vec<usize>>,
}

impl BookIndex {
    pub fn generate<R: io::Read + io::Seek>(book: &mut Epub<R>) -> Result<Self> {
        let paragraphs = combined_paragraphs(book)?;

        let model = POSModel::new(Default::default())?;
        let mut tags = HashMap::new();

        for (idx, para) in paragraphs.iter().enumerate() {
            for label in util::extract_pos_tags(para, &model) {
                tags.entry(label)
                    .and_modify(|entry: &mut Vec<usize>| entry.push(idx))
                    .or_insert_with(|| vec![idx]);
            }
        }

        Ok(Self { paragraphs, tags })
    }

    pub fn get_by_tags(&self, tags: &[String]) -> impl Iterator<Item = &str> {
        tags.iter()
            .filter_map(|tag| self.tags.get(tag).cloned())
            .flatten()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|idx| self.paragraphs.get(idx))
            .map(AsRef::as_ref)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(serde_json::from_reader(File::open(path)?)?)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        serde_json::to_writer(File::create(path)?, self)?;
        Ok(())
    }
}

const ALLOWED_TAGS: phf::Set<&'static str> = phf_set! { "a", "p" };

const CHAPTER_IGNORE_LIST: phf::Set<&'static str> = phf_set! {
  "Cover",
  "Title Page",
  "Copyright",
  "Preface",
  "Table of Contents",
  "Notes",
  "Index",
  "Introduction",
  "Foreword"
};

fn combined_paragraphs<R: io::Read + io::Seek>(book: &mut Epub<R>) -> Result<Vec<String>> {
    fn go<R: io::Read + io::Seek>(book: &mut Epub<R>, point: &NavPoint, acc: &mut Vec<String>) -> Result<()> {
        if point.children.is_empty() {
            let res = book.read(point.href().without_fragment())?;
            let doc = roxmltree::Document::parse(&res.data.0)?;

            let body = doc
                .root_element()
                .children()
                .find(|n| n.tag_name().name() == "body")
                .expect("No body in HTML");

            let paragraphs = body
                .children()
                .filter(|n| ALLOWED_TAGS.contains(n.tag_name().name()))
                .map(combined_text);

            acc.extend(paragraphs);
        } else {
            for child in &point.children {
                go(book, child, acc)?;
            }
        }
        Ok(())
    }

    let res = book.read(Href::TOC)?;
    let mut acc = vec![];

    for point in res.toc()?.points() {
        if !CHAPTER_IGNORE_LIST.contains(point.label.text.as_ref()) {
            go(book, point, &mut acc)?;
        }
    }

    Ok(acc)
}

fn combined_text(node: roxmltree::Node) -> String {
    fn go(node: roxmltree::Node, acc: &mut String) {
        if node.is_text() {
            acc.push_str(node.text().unwrap());
        } else {
            for child in node.children() {
                go(child, acc);
            }
        }
    }

    let mut acc = String::new();
    go(node, &mut acc);
    acc
}
