use phf::phf_set;
use rust_bert::pipelines::pos_tagging::POSModel;

pub fn extract_pos_tags(sentence: &str, model: &POSModel) -> Vec<String> {
    let mut tags = vec![];

    let result = model.predict([sentence]);
    let matches = result.first().unwrap();

    let groups = matches.group_by(|x, y| {
        let x: Label = x.label.as_str().into();
        let y: Label = y.label.as_str().into();

        (x == Label::NounProper && y == Label::NounProper)
            || (x == Label::Adjective && (y == Label::NounSingular || y == Label::NounPlural || y == Label::NounProper))
    });

    for group in groups {
        match group.first().unwrap().label.as_str().into() {
            Label::Other => continue,
            Label::Adjective if group.len() < 2 => continue,
            _ if group.iter().any(|tag| tag.word.chars().count() <= 2) => continue,
            _ => {}
        };

        let combined = group
            .iter()
            .map(|tag| tag.word.to_lowercase())
            .reduce(|a, b| a + " " + &b.to_lowercase())
            .unwrap();

        if !WORD_IGNORE_LIST.contains(&combined) {
            tags.push(combined);
        }
    }
    tags
}

#[derive(Debug, PartialEq, Eq)]
enum Label {
    Adjective,
    NounSingular,
    NounPlural,
    NounProper,
    Other,
}

impl<S: AsRef<str>> From<S> for Label {
    fn from(str: S) -> Self {
        match str.as_ref() {
            "JJ" => Label::Adjective,
            "NN" => Label::NounSingular,
            "NNS" => Label::NounPlural,
            "NNP" => Label::NounProper,
            _ => Label::Other,
        }
    }
}

const WORD_IGNORE_LIST: phf::Set<&'static str> = phf_set! {
  "viz",
  "view",
  "views",
  "change",
  "changes",
  "kind",
  "kinds",
  "man",
  "men",
  "thing",
  "things",
  "thinking",
  "world",
  "worlds",
  "doctrine",
  "doctrines",
  "design",
  "designs",
  "end",
  "ends",
  "mind",
  "minds",
  "fact",
  "facts",
  "realization",
  "realizations",
  "wish",
  "wishes",
  "way",
  "ways",
  "point",
  "points",
  "failure",
  "failures",
  "plan",
  "plans",
  "instance",
  "instances",
  "example",
  "examples",
  "category",
  "categories",
  "being",
  "beings",
  "account",
  "accounts",
  "response",
  "responses",
  "feasibility",
  "unfeasibility",
  "opinion",
  "opinions",
  "question",
  "questions",
  "group",
  "groups",
  "people",
  "state",
  "states",
  "scheme",
  "schemes",
  "treatment",
  "treatments",
  "something",
  "someone",
  "anything",
  "anyone",
  "problem",
  "problems",
  "condition",
  "conditions",
  "affair",
  "affairs",
  "field",
  "fields",
  "endeavor",
  "endeavors",
  "million",
  "millions",
  "year",
  "years",
  "meaning",
  "meanings",
  "work",
  "works",
  "aspect",
  "aspects",
  "event",
  "events",
  "reference",
  "references",
  "research",
  "environment",
  "object",
  "objects",
  "thought",
  "thoughts",
  "nothing",
  "process",
  "processes",
  "bearing",
  "bearings",
  "term",
  "terms",
  "basis",
  "use",
  "uses",
  "ground",
  "gorunds",
  "step",
  "steps",
  "part",
  "parts",
  "record",
  "records",
  "case",
  "cases",
  "regard",
  "course"
};
