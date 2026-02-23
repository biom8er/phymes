pub mod e_utils;
mod open_alex_author;
mod open_alex_award;
mod open_alex_common;
mod open_alex_funder;
mod open_alex_institution;
mod open_alex_publisher;
mod open_alex_request;
mod open_alex_response;
mod open_alex_source;
mod open_alex_topic;
mod open_alex_works;
pub mod semantic_scholar;

pub(crate) use open_alex_author::{
    AuthorCountsByYearTable, AuthorLastKnownInstitutionsTable, AuthorSummaryStatsTable,
};
pub(crate) use open_alex_award::{AwardAffiliationTable, AwardFunderTable};
pub(crate) use open_alex_funder::{
    FunderCountsByYearTable, FunderRoleTable, FunderSummaryStatsTable,
};
pub(crate) use open_alex_institution::{
    InstitutionCountsByYearTable, InstitutionSummaryStatsTable,
};
pub(crate) use open_alex_publisher::{
    PublisherCountsByYearTable, PublisherRoleTable, PublisherSummaryStatsTable,
};
pub(crate) use open_alex_source::{SourceCountsByYearTable, SourceSummaryStatsTable};
pub(crate) use open_alex_works::{
    WorkAwardTable, WorkCountsByYearTable, WorkFunderTable, WorkTopicTable,
};

pub mod open_alex {
    use super::*;
    pub use open_alex_author::{
        AuthorAffiliationTable, AuthorConceptTable, AuthorCountsByYearTable,
        AuthorDisplayNameAlternativesTable, AuthorIdsTable, AuthorLastKnownInstitutionsTable,
        AuthorSummaryStatsTable, AuthorTable,
    };
    pub use open_alex_award::{
        AwardAffiliationTable, AwardFundedOutputsTable, AwardFunderTable, AwardInvestigatorTable,
        AwardTable,
    };
    pub use open_alex_funder::{
        FunderAlternativeTitlesTable, FunderCountsByYearTable, FunderIdsTable, FunderRoleTable,
        FunderSummaryStatsTable, FunderTable,
    };
    pub use open_alex_institution::{
        InstitutionAssociatedInstitutionTable, InstitutionConceptTable,
        InstitutionCountsByYearTable, InstitutionDisplayNameAcronymsTable,
        InstitutionDisplayNameAlternativesTable, InstitutionGeoTable, InstitutionIdsTable,
        InstitutionInternationalNamesTable, InstitutionLineageTable, InstitutionRepositoryTable,
        InstitutionRoleTable, InstitutionSummaryStatsTable, InstitutionTable,
    };
    pub use open_alex_publisher::{
        PublisherAlternativeTitlesTable, PublisherCountryCodeTable, PublisherCountsByYearTable,
        PublisherIdsTable, PublisherLineageTable, PublisherRoleTable, PublisherSummaryStatsTable,
        PublisherTable,
    };
    pub use open_alex_request::{OPENALEX_API, OpenAlexRequest, OpenAlexRequestEntity};
    pub use open_alex_response::{
        OpenAlexResponseAuthors, OpenAlexResponseAward, OpenAlexResponseFind,
        OpenAlexResponseFunder, OpenAlexResponseGroupBy, OpenAlexResponseInstitution,
        OpenAlexResponsePublisher, OpenAlexResponseSource, OpenAlexResponseTopic,
        OpenAlexResponseWorks,
    };
    pub use open_alex_source::{
        SourceAlternativeTitlesTable, SourceApcPriceTable, SourceConceptTable,
        SourceCountsByYearTable, SourceIdsTable, SourceIssnTable, SourceLineageTable,
        SourceSocietyTable, SourceSummaryStatsTable, SourceTable,
    };
    pub use open_alex_topic::{
        TopicDomainTable, TopicFieldTable, TopicIdsTable, TopicKeywordTable, TopicSubfieldTable,
        TopicTable,
    };
    pub use open_alex_works::{
        WorkApcInfoTable, WorkAuthorshipTable, WorkAwardTable, WorkBiblioTable,
        WorkCitationPercentileTable, WorkCitedByPercentileYearTable, WorkConceptTable,
        WorkCorrespondingAuthorTable, WorkCorrespondingInstitutionTable, WorkCountsByYearTable,
        WorkFunderTable, WorkIdsTable, WorkIndexedInTable, WorkKeywordTable, WorkLocationTable,
        WorkMeshTagTable, WorkOpenAccessTable, WorkReferencedWorksTable, WorkRelatedWorksTable,
        WorkSdgTagTable, WorkTable, WorkTopicTable,
    };
}
