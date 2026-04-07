use std::fmt::Display;

use crate::http::{
    AuthorCountsByYearTable, AuthorSummaryStatsTable, FunderCountsByYearTable,
    FunderSummaryStatsTable, InstitutionCountsByYearTable, InstitutionSummaryStatsTable,
    PublisherCountsByYearTable, PublisherSummaryStatsTable, SourceCountsByYearTable,
    SourceSummaryStatsTable, WorkCountsByYearTable,
    open_alex_author::Author,
    open_alex_award::Award,
    open_alex_funder::Funder,
    open_alex_institution::{Geo, Institution},
    open_alex_publisher::Publisher,
    open_alex_source::Source,
    open_alex_topic::Topic,
    open_alex_works::Work,
};
use clap::ValueEnum;
use phymes_diagnostics::HashMap;
use serde::{Deserialize, Serialize};

/// OpenAlex Entities
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OpenAlexEntity {
    Work(Work),
    Author(Author),
    Source(Source),
    Institution(Institution),
    Topic(Topic),
    Keyword(Keyword),
    Publisher(Publisher),
    Funder(Funder),
    Award(Award),
    Geo(Geo),
    Concept(Concept),
    #[default]
    #[serde(other)]
    Unknown,
}

//
// ===== Shared enums =====
//

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum CountryCode {
    US,
    GB,
    DE,
    FR,
    NL,
    CN,
    JP,
    IN,
    ES,
    IT,
    #[default]
    #[serde(other)]
    Unknown,
}

impl Display for CountryCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::US => write!(f, "US"),
            Self::GB => write!(f, "GB"),
            Self::DE => write!(f, "DE"),
            Self::FR => write!(f, "FR"),
            Self::NL => write!(f, "NL"),
            Self::CN => write!(f, "CN"),
            Self::JP => write!(f, "JP"),
            Self::IN => write!(f, "IN"),
            Self::ES => write!(f, "ES"),
            Self::IT => write!(f, "IT"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum Currency {
    USD,
    EUR,
    GBP,
    JPY,
    CNY,
    AUD,
    CAD,
    #[default]
    #[serde(other)]
    Unknown,
}

impl Display for Currency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::USD => write!(f, "USD"),
            Self::EUR => write!(f, "EUR"),
            Self::GBP => write!(f, "GBP"),
            Self::JPY => write!(f, "JPY"),
            Self::CNY => write!(f, "CNY"),
            Self::AUD => write!(f, "AUD"),
            Self::CAD => write!(f, "CAD"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ConceptLevel {
    Domain,
    Field,
    Subfield,
    Topic,
    #[default]
    #[serde(other)]
    Unknown,
}

impl Display for ConceptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Domain => write!(f, "Domain"),
            Self::Field => write!(f, "Field"),
            Self::Subfield => write!(f, "Subfield"),
            Self::Topic => write!(f, "JPY"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum InstitutionRelationship {
    Parent,
    Child,
    Related,
    #[default]
    #[serde(other)]
    Unknown,
}

impl Display for InstitutionRelationship {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parent => write!(f, "Parent"),
            Self::Child => write!(f, "Child"),
            Self::Related => write!(f, "Related"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum InstitutionType {
    Archive,
    Company,
    Education,
    Facility,
    Government,
    Healthcare,
    Nonprofit,
    #[default]
    #[serde(other)]
    Other,
}

impl Display for InstitutionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Archive => write!(f, "Archive"),
            Self::Company => write!(f, "Company"),
            Self::Education => write!(f, "Education"),
            Self::Facility => write!(f, "Facility"),
            Self::Government => write!(f, "Government"),
            Self::Healthcare => write!(f, "Healthcare"),
            Self::Nonprofit => write!(f, "Nonprofit"),
            Self::Other => write!(f, "Other"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum RoleType {
    Institution,
    Funder,
    Publisher,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    Journal,
    Repository,
    Conference,
    EbookPlatform,
    BookSeries,
    Metadata,
    Other,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum KeywordType {
    Phrase,
    Term,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq, ValueEnum)]
pub enum WorkType {
    #[value(name = "Article")]
    #[serde(alias = "article")]
    Article,
    /// Added to support peer review
    #[value(name = "AuthorResponse")]
    #[serde(alias = "author-response")]
    AuthorResponse,
    #[value(name = "Book")]
    #[serde(alias = "book")]
    Book,
    #[value(name = "BookChapter")]
    #[serde(alias = "book-chapter")]
    BookChapter,
    #[value(name = "Dataset")]
    #[serde(alias = "dataset")]
    Dataset,
    #[value(name = "Dissertation")]
    #[serde(alias = "dissertation")]
    Dissertation,
    #[value(name = "Editorial")]
    #[serde(alias = "editorial")]
    Editorial,
    /// Added to support peer review
    #[value(name = "EditorDecision")]
    #[serde(alias = "editor-decision")]
    EditorDecision,
    #[value(name = "Erratum")]
    #[serde(alias = "erratum")]
    Erratum,
    #[value(name = "Letter")]
    #[serde(alias = "letter")]
    Letter,
    #[value(name = "Libguides")]
    #[serde(alias = "libguides")]
    Libguides,
    #[value(name = "Paratext")]
    #[serde(alias = "paratext")]
    Paratext,
    /// Also called RefereeReport
    #[value(name = "PeerReview")]
    #[serde(alias = "peer-review")]
    PeerReview,
    #[value(name = "Preprint")]
    #[serde(alias = "preprint")]
    Preprint,
    #[value(name = "ReferenceEntry")]
    #[serde(alias = "reference-entry")]
    ReferenceEntry,
    #[value(name = "Report")]
    #[serde(alias = "report")]
    Report,
    #[value(name = "Retraction")]
    #[serde(alias = "retraction")]
    Retraction,
    #[value(name = "Review")]
    #[serde(alias = "review")]
    Review,
    #[value(name = "Standard")]
    #[serde(alias = "standard")]
    Standard,
    #[value(name = "SupplementaryMaterials")]
    #[serde(alias = "supplementary-materials")]
    SupplementaryMaterials,
    #[value(name = "Other")]
    #[serde(alias = "other")]
    #[default]
    #[serde(other)]
    Other,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OaStatus {
    Gold,
    Hybrid,
    Green,
    Bronze,
    Closed,
    #[default]
    #[serde(other)]
    Unknown,
}

impl Display for OaStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gold => write!(f, "Gold"),
            Self::Hybrid => write!(f, "Hybrid"),
            Self::Green => write!(f, "Green"),
            Self::Bronze => write!(f, "Bronze"),
            Self::Closed => write!(f, "Closed"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LanguageCode {
    En,
    De,
    Fr,
    Es,
    Zh,
    Ja,
    Ru,
    Pt,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AuthorPosition {
    First,
    Middle,
    Last,
    Solo,
    #[default]
    #[serde(other)]
    Unknown,
}

impl Display for AuthorPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::First => write!(f, "First"),
            Self::Middle => write!(f, "Middle"),
            Self::Last => write!(f, "Last"),
            Self::Solo => write!(f, "Solo"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

//
// ===== Shared structs =====
//

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct CountsByYear {
    pub year: u32,
    pub works_count: Option<u32>,
    pub cited_by_count: u32,
}

impl CountsByYear {
    pub fn build_work_counts_by_year(self, work_id: &str) -> WorkCountsByYearTable {
        WorkCountsByYearTable {
            work_id: work_id.to_string(),
            year: self.year,
            cited_by_count: self.cited_by_count,
        }
    }
    pub fn build_author_counts_by_year(self, author_id: &str) -> AuthorCountsByYearTable {
        AuthorCountsByYearTable {
            author_id: author_id.to_string(),
            year: self.year,
            cited_by_count: self.cited_by_count,
        }
    }
    pub fn build_institution_counts_by_year(
        self,
        institution_id: &str,
    ) -> InstitutionCountsByYearTable {
        InstitutionCountsByYearTable {
            institution_id: institution_id.to_string(),
            year: self.year,
            cited_by_count: self.cited_by_count,
        }
    }
    pub fn build_funder_counts_by_year(self, funder_id: &str) -> FunderCountsByYearTable {
        FunderCountsByYearTable {
            funder_id: funder_id.to_string(),
            year: self.year,
            cited_by_count: self.cited_by_count,
        }
    }
    pub fn build_publisher_counts_by_year(self, publisher_id: &str) -> PublisherCountsByYearTable {
        PublisherCountsByYearTable {
            publisher_id: publisher_id.to_string(),
            year: self.year,
            cited_by_count: self.cited_by_count,
        }
    }
    pub fn build_source_counts_by_year(self, source_id: &str) -> SourceCountsByYearTable {
        SourceCountsByYearTable {
            source_id: source_id.to_string(),
            year: self.year,
            cited_by_count: self.cited_by_count,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct SummaryStats {
    #[serde(rename = "2yr_mean_citedness")]
    pub two_year_mean_citedness: Option<f64>,
    pub h_index: Option<u32>,
    pub i10_index: Option<u32>,
}

impl SummaryStats {
    pub fn build_author_summary_stats_table(self, author_id: &str) -> AuthorSummaryStatsTable {
        AuthorSummaryStatsTable {
            author_id: author_id.to_string(),
            two_year_mean_citedness: self.two_year_mean_citedness.unwrap_or_default(),
            h_index: self.h_index.unwrap_or_default(),
            i10_index: self.i10_index.unwrap_or_default(),
        }
    }
    pub fn build_institution_summary_stats_table(
        self,
        institution_id: &str,
    ) -> InstitutionSummaryStatsTable {
        InstitutionSummaryStatsTable {
            institution_id: institution_id.to_string(),
            two_year_mean_citedness: self.two_year_mean_citedness.unwrap_or_default(),
            h_index: self.h_index.unwrap_or_default(),
            i10_index: self.i10_index.unwrap_or_default(),
        }
    }
    pub fn build_funder_summary_stats_table(self, funder_id: &str) -> FunderSummaryStatsTable {
        FunderSummaryStatsTable {
            funder_id: funder_id.to_string(),
            two_year_mean_citedness: self.two_year_mean_citedness.unwrap_or_default(),
            h_index: self.h_index.unwrap_or_default(),
            i10_index: self.i10_index.unwrap_or_default(),
        }
    }
    pub fn build_publisher_summary_stats_table(
        self,
        publisher_id: &str,
    ) -> PublisherSummaryStatsTable {
        PublisherSummaryStatsTable {
            publisher_id: publisher_id.to_string(),
            two_year_mean_citedness: self.two_year_mean_citedness.unwrap_or_default(),
            h_index: self.h_index.unwrap_or_default(),
            i10_index: self.i10_index.unwrap_or_default(),
        }
    }
    pub fn build_source_summary_stats_table(self, source_id: &str) -> SourceSummaryStatsTable {
        SourceSummaryStatsTable {
            source_id: source_id.to_string(),
            two_year_mean_citedness: self.two_year_mean_citedness.unwrap_or_default(),
            h_index: self.h_index.unwrap_or_default(),
            i10_index: self.i10_index.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Concept {
    pub id: Option<String>,
    pub wikidata: Option<String>,
    pub display_name: Option<String>,
    // pub level: Option<ConceptLevel>,
    pub level: Option<u32>,
    pub score: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Keyword {
    pub id: Option<String>,
    pub display_name: Option<String>,
    pub created_date: Option<String>,
    pub updated_date: Option<String>,
    pub cited_by_count: Option<u32>,
    pub works_count: Option<u32>,
    #[serde(rename = "type")]
    pub type_: Option<KeywordType>,
}

// Documentation for the OpenAlex API in MarkDown from <https://docs.openalex.org/>
#[allow(dead_code)]
const OPENALEX_API_DOCUMENTATION: &str = r#"# Get lists of entities

It's easy to get a list of entity objects from from the API:`/<entity_name>`. Here's an example:

* Get a list of *all* the topics in OpenAlex:\
  [`https://api.openalex.org/topics`](https://api.openalex.org/topics)

This query returns a `meta` object with details about the query, a `results` list of [`Topic`](https://docs.openalex.org/api-entities/topics/topic-object) objects, and an empty [`group_by`](https://docs.openalex.org/how-to-use-the-api/get-groups-of-entities) list:

```json
meta: {
    count: 4516,
    db_response_time_ms: 81,
    page: 1,
    per_page: 25
    },
results: [
    // long list of Topic entities
 ],
group_by: [] // empty
```

Listing entities is a lot more useful when you add parameters to [page](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/paging), [filter](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists), [search](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities), and [sort](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/sort-entity-lists) them. Keep reading to learn how to do that.

# Paging

{% hint style="info" %}
You can see executable examples of paging in [this user-contributed Jupyter notebook!](https://github.com/ourresearch/openalex-api-tutorials/blob/main/notebooks/getting-started/paging.ipynb)
{% endhint %}

### Basic paging

Use the `page` query parameter to control which page of results you want (eg `page=1`, `page=2`, etc). By default there are 25 results per page; you can use the `per-page` parameter to change that to any number between 1 and 200.

* Get the 2nd page of a list:\
  [`https://api.openalex.org/works?page=2`](https://api.openalex.org/works?page=2)
* Get 200 results on the second page:\
  [`https://api.openalex.org/works?page=2&per-page=200`](https://api.openalex.org/works?page=2\&per-page=200)

Basic paging only works to get the first 10,000 results of any list. If you want to see more than 10,000 results, you'll need to use [cursor paging](#cursor-paging).

### Cursor paging

Cursor paging is a bit more complicated than [basic paging](#basic-paging), but it allows you to access as many records as you like.

To use cursor paging, you request a cursor by adding the `cursor=*` parameter-value pair to your query.

* Get a cursor in order to start cursor pagination:\
  [`https://api.openalex.org/works?filter=publication_year:2020&per-page=100&cursor=*`](https://api.openalex.org/works?filter=publication_year:2020\&per-page=100\&cursor=*)

The response to your query will include a `next_cursor` value in the response's `meta` object. Here's what it looks like:

```json
{
  "meta": {
    "count": 8695857,
    "db_response_time_ms": 28,
    "page": null,
    "per_page": 100,
    "next_cursor": "IlsxNjA5MzcyODAwMDAwLCAnaHR0cHM6Ly9vcGVuYWxleC5vcmcvVzI0ODg0OTk3NjQnXSI="
  },
  "results" : [
    // the first page of results
  ]
}
```

To retrieve the next page of results, copy the `meta.next_cursor` value into the cursor field of your next request.

* Get the next page of results using a cursor value:\
  [`https://api.openalex.org/works?filter=publication_year:2020&per-page=100&cursor=IlsxNjA5MzcyODAwMDAwLCAnaHR0cHM6Ly9vcGVuYWxleC5vcmcvVzI0ODg0OTk3NjQnXSI=`](https://api.openalex.org/works?filter=publication_year:2020\&per-page=100\&cursor=IlsxNjA5MzcyODAwMDAwLCAnaHR0cHM6Ly9vcGVuYWxleC5vcmcvVzI0ODg0OTk3NjQnXSI=)

This second page of results will have a new value for `meta.next_cursor`. You'll use this new value the same way you did the first, and it'll give you the second page of results. To get *all* the results, keep repeating this process until `meta.next_cursor` is null and the `results` set is empty.

Besides using cursor paging to get entities, you can also use it in [`group_by` queries](https://docs.openalex.org/how-to-use-the-api/get-groups-of-entities).

{% hint style="danger" %}
**Don't use cursor paging to download the whole dataset.**

* It's bad for you because it will take many days to page through a long list like /works or /authors.
* It's bad for us (and other users!) because it puts a massive load on our servers.

Instead, download everything at once, using the [OpenAlex snapshot](https://docs.openalex.org/download-all-data/openalex-snapshot). It's free, easy, fast, and you get all the results in same format you'd get from the API.
{% endhint %}

# Filter entity lists

Filters narrow the list down to just entities that meet a particular condition--specifically, a particular value for a particular attribute.

A list of filters are set using the `filter` parameter, formatted like this: `filter=attribute:value,attribute2:value2`. Examples:

* Get the works whose [type](https://docs.openalex.org/api-entities/works/work-object#type) is `book`:\
  [`https://api.openalex.org/works?filter=type:book`](https://api.openalex.org/works?filter=type:book)
* Get the authors whose name is Einstein:\
  [`https://api.openalex.org/authors?filter=display_name.search:einstein`](https://api.openalex.org/authors?filter=display_name.search:einstein)

Filters are case-insensitive.

## Logical expressions

### Inequality

For numerical filters, use the less-than (`<`) and greater-than (`>`) symbols to filter by inequalities. Example:

* Get sources that host more than 1000 works:\
  [`https://api.openalex.org/sources?filter=works_count:>1000`](https://api.openalex.org/sources?filter=works_count:%3E1000)

Some attributes have special filters that act as syntactic sugar around commonly-expressed inequalities: for example, the `from_publication_date` filter on `works`. See the endpoint-specific documentation below for more information. Example:

* Get all works published between 2022-01-01 and 2022-01-26 (inclusive):\
  [`https://api.openalex.org/works?filter=from_publication_date:2022-01-01,to_publication_date:2022-01-26`](https://api.openalex.org/works?filter=from_publication_date:2022-01-01,to_publication_date:2022-01-26)

### Negation (NOT)

You can negate any filter, numerical or otherwise, by prepending the exclamation mark symbol (`!`) to the filter value. Example:

* Get all institutions *except* for ones located in the US:\
  [`https://api.openalex.org/institutions?filter=country_code:!us`](https://api.openalex.org/institutions?filter=country_code:!us)

### Intersection (AND)

By default, the returned result set includes only records that satisfy *all* the supplied filters. In other words, filters are combined as an AND query. Example:

* Get all works that have been cited more than once *and* are free to read:\
  [`https://api.openalex.org/works?filter=cited_by_count:>1,is_oa:true`](https://api.openalex.org/works?filter=cited_by_count:%3E1,is_oa:true)

To create an AND query within a single attribute, you can either repeat a filter, or use the plus symbol (`+`):

* Get all the works that have an author from France *and* an author from the UK:
  * Using repeating filters: [`https://api.openalex.org/works?filter=institutions.country_code:fr,institutions.country_code:gb`](https://api.openalex.org/works?filter=institutions.country_code:fr,institutions.country_code:gb)
  * Using the plus symbol (`+`): [`https://api.openalex.org/works?filter=institutions.country_code:fr+gb`](https://api.openalex.org/works?filter=institutions.country_code:fr+gb)

Note that the plus symbol (`+`) syntax will not work for search filters, boolean filters, or numeric filters.

### Addition (OR)

Use the pipe symbol (`|`) to input lists of values such that *any* of the values can be satisfied--in other words, when you separate filter values with a pipe, they'll be combined as an `OR` query. Example:

* Get all the works that have an author from France or an author from the UK:\
  [`https://api.openalex.org/works?filter=institutions.country_code:fr|gb`](https://api.openalex.org/works?filter=institutions.country_code:fr|gb)

This is particularly useful when you want to retrieve a many records by ID all at once. Instead of making a whole bunch of singleton calls in a loop, you can make one call, like this:

* Get the works with DOI `10.1371/journal.pone.0266781` *or* with DOI `10.1371/journal.pone.0267149` (note the pipe separator between the two DOIs):\
  [`https://api.openalex.org/works?filter=doi:https://doi.org/10.1371/journal.pone.0266781|https://doi.org/10.1371/journal.pone.0267149`](https://api.openalex.org/works?filter=doi:https://doi.org/10.1371/journal.pone.0266781|https://doi.org/10.1371/journal.pone.0267149)

You can combine up to 100 values for a given filter in this way. You will also need to use the parameter `per-page=100` to get all of the results per query. See our [blog post](https://blog.ourresearch.org/fetch-multiple-dois-in-one-openalex-api-request/) for a tutorial.

{% hint style="danger" %}
You can use OR for values *within* a given filter, but not *between* different filters. So this, for example, **doesn't work and will return an error**:

* Get either French works *or* ones published in the journal with ISSN 0957-1558:\
  [`https://api.openalex.org/works?filter=institutions.country_code:fr|primary_location.source.issn:0957-1558`](https://api.openalex.org/works?filter=institutions.country_code:fr|primary_location.source.issn:0957-1558)
  {% endhint %}

## Available Filters

The filters for each entity can be found here:

* [Works](https://docs.openalex.org/api-entities/works/filter-works)
* [Authors](https://docs.openalex.org/api-entities/authors/filter-authors)
* [Sources](https://docs.openalex.org/api-entities/sources/filter-sources)
* [Institutions](https://docs.openalex.org/api-entities/institutions/filter-institutions)
* [Concepts](https://docs.openalex.org/api-entities/concepts/filter-concepts)
* [Publishers](https://docs.openalex.org/api-entities/publishers/filter-publishers)
* [Funders](https://docs.openalex.org/api-entities/funders/filter-funders)

{% hint style="info" %}
**Looking for text search?** Filters match exact attribute values. If you want to search for words in titles, abstracts, or other text fields, see [Search entities](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/search-entities). Or, for AI-powered semantic search that finds conceptually related works even when they use different terminology, check out [Find similar works](https://docs.openalex.org/how-to-use-the-api/find-similar-works).
{% endhint %}

# Search entities

## The `search` parameter

The `search` query parameter finds results that match a given text search. Example:

* Get works with search term "dna" in the title, abstract, or fulltext:\
  [`https://api.openalex.org/works?search=dna`](https://api.openalex.org/works?search=dna)

When you [search `works`](https://docs.openalex.org/api-entities/works/search-works), the API looks for matches in titles, abstracts, and [fulltext](https://docs.openalex.org/api-entities/works/work-object#has_fulltext). When you [search `concepts`](https://docs.openalex.org/api-entities/concepts/search-concepts), we look in each concept's `display_name` and `description` fields. When you [search `sources`](https://docs.openalex.org/api-entities/sources/search-sources), we look at the `display_name`*,* `alternate_titles`, and `abbreviated_title` fields. When you [search `authors`](https://docs.openalex.org/api-entities/authors/search-authors), we look at the `display_name` and `display_name_alternatives` fields. When you [search `institutions`](https://docs.openalex.org/api-entities/institutions/search-institutions), we look at the `display_name`, `display_name_alternatives`, and `display_name_acronyms` fields.

For most text search we remove [stop words](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-stop-tokenfilter.html) and use [stemming](https://en.wikipedia.org/wiki/Stemming) (specifically, the [Kstem token filter](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-kstem-tokenfilter.html)) to improve results. So words like "the" and "an" are transparently removed, and a search for "possums" will also return records using the word "possum." With the exception of raw affiliation strings, we do not search within words but rather try to match whole words. So a search with "lun" will not match the word "lunar".

### Search without stemming

To disable stemming and the removal of stop words for searches on titles and abstracts, you can add `.no_stem` to the search filter. So, for example, if you want to search for "surgery" and not get "surgeries" too:

* [`https://api.openalex.org/works?filter=display_name.search.no_stem:surgery`](https://api.openalex.org/works?filter=display_name.search.no_stem:surgery)
* [`https://api.openalex.org/works?filter=title.search.no_stem:surgery`](https://api.openalex.org/works?filter=title.search.no_stem:surgery)
* [`https://api.openalex.org/works?filter=abstract.search.no_stem:surgery`](https://api.openalex.org/works?filter=abstract.search.no_stem:surgery)
* [`https://api.openalex.org/works?filter=title_and_abstract.search.no_stem:surgery`](https://api.openalex.org/works?filter=title_and_abstract.search.no_stem:surgery)

### Boolean searches

Including any of the words `AND`, `OR`, or `NOT` in any of your searches will enable boolean search. Those words must be UPPERCASE. You can use this in all searches, including using the `search` parameter, and using [search filters](#the-search-filter).

This allows you to craft complex queries using those boolean operators along with parentheses and quotation marks. Surrounding a phrase with quotation marks will search for an exact match of that phrase, after stemming and stop-word removal (be sure to use **double quotation marks** — `"`). Using parentheses will specify order of operations for the boolean operators. Words that are not separated by one of the boolean operators will be interpreted as `AND`.

Behind the scenes, the boolean search is using Elasticsearch's [query string query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-query-string-query.html) on the searchable fields (such as title, abstract, and fulltext for works; see each individual entity page for specifics about that entity). Wildcard and fuzzy searches using `*`, `?` or `~` are not allowed; these characters will be removed from any searches. These searches, even when using quotation marks, will go through the same cleaning as desscribed above, including stemming and removal of stop words.

* Search for works that mention "elmo" and "sesame street," but not the words "cookie" or "monster": [`https://api.openalex.org/works?search=(elmo AND "sesame street") NOT (cookie OR monster)`](https://api.openalex.org/works?search=%28elmo%20AND%20%22sesame%20street%22%29%20NOT%20%28cookie%20OR%20monster%29)

## Relevance score

When you use search, each returned entity in the results lists gets an extra property called `relevance_score`, and the list is by default sorted in descending order of `relevance_score`. The `relevance_score` is based on text similarity to your search term. It also includes a weighting term for citation counts: more highly-cited entities score higher, all else being equal.

If you search for a multiple-word phrase, the algorithm will treat each word separately, and rank results higher when the words appear close together. If you want to return only results where the exact phrase is used, just enclose your phrase within quotes. Example:

* Get works with the exact phrase "fierce creatures" in the title or abstract (returns just a few results):\
  [`https://api.openalex.org/works?search="fierce%20creatures"`](https://api.openalex.org/works?search=%22fierce%20creatures%22)
* Get works with the words "fierce" and "creatures" in the title or abstract, with works that have the two words close together ranked higher by `relevance_score` (returns way more results):\
  [`https://api.openalex.org/works?search=fierce%20creatures`](https://api.openalex.org/works?search=fierce%20creatures)

## The search filter

You can also use search as a [filter](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists), allowing you to fine-tune the fields you're searching over. To do this, you append `.search` to the end of the property you are filtering for:

* Get authors who have "Einstein" as part of their name:\
  [`https://api.openalex.org/authors?filter=display_name.search:einstein`](https://api.openalex.org/authors?filter=display_name.search:einstein)
* Get works with "cubist" in the title:\
  [`https://api.openalex.org/works?filter=title.search:cubist`](https://api.openalex.org/works?filter=title.search:cubist)

Additionally, the filter `default.search` is available on all entities; this works the same as the [`search` parameter](#the-search-parameter).

{% hint style="info" %}
You might be tempted to use the search filter to power an autocomplete or typeahead. Instead, we recommend you use the [autocomplete endpoint](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/autocomplete-entities), which is much faster.\
\
👎 [`https://api.openalex.org/institutions?filter=display_name.search:florida`](https://api.openalex.org/institutions?filter=display_name.search:florida)

👍 [`https://api.openalex.org/autocomplete/institutions?q=Florida`](https://api.openalex.org/autocomplete/institutions?q=Florida)
{% endhint %}

## Keyword search vs. semantic search

The keyword search described on this page finds works containing specific words or phrases. Use it when you know the exact terminology you're looking for, or when you need to combine search with other filters and sorting options.

If you want to find works that are *conceptually similar*—even when they use different terminology—use [Find similar works](https://docs.openalex.org/how-to-use-the-api/find-similar-works) instead. Semantic search uses AI embeddings to match by meaning, so a query about "machine learning in healthcare" will find relevant papers even if they use terms like "AI-driven diagnosis" or "computational medicine."

| Use keyword search when...               | Use semantic search when...           |
| ---------------------------------------- | ------------------------------------- |
| You know the exact terms to search for   | You want conceptually related works   |
| You need to combine with filters/sorting | You're exploring a new research area  |
| You want to search specific fields       | Your query is a sentence or paragraph |

# Sort entity lists

Use the `?sort` parameter to specify the property you want your list sorted by. You can sort by these properties, where they exist:

* `display_name`
* `cited_by_count`
* `works_count`
* `publication_date`
* `relevance_score` (only exists if there's a [search filter](#search) active)

By default, sort direction is ascending. You can reverse this by appending `:desc` to the sort key like `works_count:desc`. You can sort by multiple properties by providing multiple sort keys, separated by commas. Examples:

* All works, sorted by `cited_by_count` (highest counts first)\
  [`https://api.openalex.org/works?sort=cited_by_count:desc`](https://api.openalex.org/works?sort=cited_by_count:desc)
* All sources, in alphabetical order by title:\
  [`https://api.openalex.org/sources?sort=display_name`](https://api.openalex.org/sources?sort=display_name)

You can sort by relevance\_score when searching:

* Sort by year, then by relevance\_score when searching for "bioplastics":\
  [`https://api.openalex.org/works?filter=display_name.search:bioplastics&sort=publication_year:desc,relevance_score:desc`](https://api.openalex.org/works?filter=display_name.search:bioplastics\&sort=publication_year:desc,relevance_score:desc)

An error is thrown if attempting to sort by `relevance_score` without a search query.

# Select fields

You can use `select` to limit the fields that are returned in results.

* Display works with only the `id`, `doi`, and `display_name` returned in the results\
  [`https://api.openalex.org/works?select=id,doi,display\_name`](https://api.openalex.org/works?select=id,doi,display_name)

```json
"results": [
  {
    "id": "https://openalex.org/W1775749144",
    "doi": "https://doi.org/10.1016/s0021-9258(19)52451-6",
    "display_name": "PROTEIN MEASUREMENT WITH THE FOLIN PHENOL REAGENT"
  },
  {
    "id": "https://openalex.org/W2100837269",
    "doi": "https://doi.org/10.1038/227680a0",
    "display_name": "Cleavage of Structural Proteins during the Assembly of the Head of Bacteriophage T4"
  },
  // more results removed for brevity
]
```

## Limitations

The fields you choose must exist within the entity (of course). You can only select root-level fields.

So if we have a record like so:

```
"id": "https://openalex.org/W2138270253",
"open_access": {
  "is_oa": true,
  "oa_status": "bronze",
  "oa_url": "http://www.pnas.org/content/74/12/5463.full.pdf"
}
```

You can choose to display `id` and `open_access`, but you will get an error if you try to choose `open_access.is_oa`.

You can use select fields when getting lists of entities or a [single entity](https://docs.openalex.org/how-to-use-the-api/get-single-entities/select-fields). It does not work with [group-by](https://docs.openalex.org/how-to-use-the-api/get-groups-of-entities) or [autocomplete](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/autocomplete-entities).

# Sample entity lists

You can use `sample` to get a random list of up to 10,000 results.

* Get 100 random works\
  <https://api.openalex.org/works?sample=100&per-page=100>
* Get 50 random works that are open access and published in 2021\
  <https://api.openalex.org/works?filter=open_access.is_oa:true,publication_year:2021&sample=50&per-page=50>

You can add a `seed` value in order to retrieve the same set of random records, in the same order, multiple times.

* Get 20 random sources with a seed value\
  <https://api.openalex.org/sources?sample=20&seed=123>

{% hint style="info" %}
Depending on your query, random results with a seed value *may* change over time due to new records coming into OpenAlex.
{% endhint %}

## Limitations

* The sample size is limited to 10,000 results.
* You must provide a `seed` value when paging beyond the first page of results. Without a seed value, you might get duplicate records in your results.
* You must use [basic paging](https://docs.openalex.org/how-to-use-the-api/paging#basic-paging) when sampling. Cursor pagination is not supported.
"#;

use std::borrow::Borrow;

pub fn abstract_from_inverted_index<K>(inverted: &HashMap<K, Vec<usize>>) -> String
where
    K: Borrow<str>,
{
    // Find the maximum index to size the vector
    let max_index = inverted
        .values()
        .flat_map(|positions| positions.iter())
        .copied()
        .max()
        .unwrap_or(0);

    // Preallocate vector of words
    let mut words = vec![String::new(); max_index + 1];

    // Fill in words at their positions
    for (key, positions) in inverted {
        let word = key.borrow();
        for &pos in positions {
            if pos < words.len() {
                words[pos] = word.to_owned();
            }
        }
    }

    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn test_abstract_from_inverted_index_string_keys() {
        let mut idx: HashMap<String, Vec<usize>> = HashMap::new();
        idx.insert("hello".into(), vec![0]);
        idx.insert("world".into(), vec![1]);

        assert_eq!(abstract_from_inverted_index(&idx), "hello world");
    }

    #[test]
    fn test_abstract_from_inverted_index_str_keys() {
        let mut idx: HashMap<&str, Vec<usize>> = HashMap::new();
        idx.insert("foo", vec![0]);
        idx.insert("bar", vec![1]);

        assert_eq!(abstract_from_inverted_index(&idx), "foo bar");
    }

    #[test]
    fn test_abstract_from_inverted_index_cow_keys() {
        let mut idx: HashMap<Cow<'static, str>, Vec<usize>> = HashMap::new();
        idx.insert(Cow::Borrowed("alpha"), vec![1]);
        idx.insert(Cow::Owned("beta".into()), vec![0]);

        assert_eq!(abstract_from_inverted_index(&idx), "beta alpha");
    }
}
