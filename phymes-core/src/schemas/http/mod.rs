pub mod e_utils;
mod open_alex_common;
mod open_alex_request;
mod open_alex_response;
mod open_alex_works;
mod open_alex_author;
mod open_alex_funder;
mod open_alex_award;
mod open_alex_publisher;
mod open_alex_topic;
mod open_alex_institution;
mod open_alex_source;
pub mod semantic_scholar;

pub(crate) use open_alex_works::{WorkAwardTable, WorkFunderTable, WorkCountsByYearTable, WorkTopicTable};
pub(crate) use open_alex_author::{AuthorLastKnownInstitutionsTable, AuthorSummaryStatsTable, AuthorCountsByYearTable};
pub(crate) use open_alex_funder::{FunderRoleTable, FunderCountsByYearTable, FunderSummaryStatsTable};
pub(crate) use open_alex_award::{AwardFunderTable, AwardAffiliationTable};
pub(crate) use open_alex_publisher::{PublisherRoleTable, PublisherCountsByYearTable, PublisherSummaryStatsTable};
pub(crate) use open_alex_institution::{InstitutionSummaryStatsTable, InstitutionCountsByYearTable};
pub(crate) use open_alex_source::{SourceCountsByYearTable, SourceSummaryStatsTable};

pub mod open_alex {
    use super::*;
    pub use open_alex_request::{OPENALEX_API, OpenAlexRequest, OpenAlexRequestEntity};
    pub use open_alex_response::{OpenAlexResponseWorks, OpenAlexResponseAuthors, OpenAlexResponseInstitution, OpenAlexResponseTopic, OpenAlexResponseFind, OpenAlexResponseGroupBy};
    pub use open_alex_works::{WorkTable, WorkAuthorshipTable, WorkApcInfoTable, WorkAwardTable, WorkFunderTable, WorkLocationTable, WorkOpenAccessTable, 
        WorkBiblioTable, WorkCitationPercentileTable, WorkCitedByPercentileYearTable, WorkCountsByYearTable, WorkKeywordTable, WorkMeshTagTable, WorkSdgTagTable, 
        WorkIdsTable, WorkTopicTable, WorkConceptTable, WorkCorrespondingAuthorTable, WorkCorrespondingInstitutionTable, WorkIndexedInTable, WorkReferencedWorksTable, 
        WorkRelatedWorksTable};
    pub use open_alex_author::{AuthorTable, AuthorDisplayNameAlternativesTable, AuthorAffiliationTable, AuthorLastKnownInstitutionsTable, AuthorIdsTable, AuthorSummaryStatsTable, AuthorCountsByYearTable, AuthorConceptTable};
    pub use open_alex_funder::{FunderTable, FunderAlternativeTitlesTable, FunderIdsTable, FunderRoleTable, FunderCountsByYearTable, FunderSummaryStatsTable};
    pub use open_alex_award::{AwardTable, AwardFunderTable, AwardFundedOutputsTable, AwardInvestigatorTable, AwardAffiliationTable};
    pub use open_alex_publisher::{PublisherTable, PublisherAlternativeTitlesTable, PublisherCountryCodeTable, PublisherLineageTable, PublisherIdsTable, PublisherRoleTable, PublisherCountsByYearTable, PublisherSummaryStatsTable};
    pub use open_alex_topic::{TopicTable, TopicDomainTable, TopicFieldTable, TopicSubfieldTable, TopicIdsTable, TopicKeywordTable};
    pub use open_alex_institution::{InstitutionTable, InstitutionDisplayNameAcronymsTable, InstitutionDisplayNameAlternativesTable, InstitutionGeoTable, 
        InstitutionIdsTable, InstitutionAssociatedInstitutionTable, InstitutionRepositoryTable, InstitutionRoleTable, InstitutionInternationalNamesTable, 
        InstitutionSummaryStatsTable, InstitutionCountsByYearTable, InstitutionConceptTable, InstitutionLineageTable};
    pub use open_alex_source::{SourceTable, SourceAlternativeTitlesTable, SourceApcPriceTable, SourceCountsByYearTable, SourceLineageTable, SourceIdsTable, 
        SourceIssnTable, SourceSocietyTable, SourceSummaryStatsTable, SourceConceptTable};
}
