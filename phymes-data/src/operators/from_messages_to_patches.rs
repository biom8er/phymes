use anyhow::{anyhow, Result};
use arrow::array::RecordBatch;
use candle_core::Device;
use phymes_schemas::create_workspace_patch_batch;
use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
};
use serde::{Deserialize, Serialize};

use crate::{CodeCompletionType, DataConfig, DataOperatorTrait, PatchOperator, parse_fill_in_the_middle_output, parse_search_and_replace_output};

/// Compute the normalized start and end times in a [RecordBatch]
#[derive(Debug, Serialize, Deserialize)]
pub struct FromMessagesToPatches {
    code_completion: CodeCompletionType,
}

impl MappableTrait for FromMessagesToPatches {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}

impl DataOperatorTrait for FromMessagesToPatches {
    fn forward(
        &self,
        lhs_args: &[RecordBatch],
        _rhs_args: Option<&[RecordBatch]>,
        device: &Device,
    ) -> Result<RecordBatch> {
        from_messages_to_patches(lhs_args, &self.code_completion, device)
    }
    fn new(config: &DataConfig) -> Result<Self> {
        let code_completion = config.code_completion.clone().ok_or(anyhow!(
            "Missing `code_completion` for `{}`.",
            Self::get_static_name()
        ))?;
        Ok(FromMessagesToPatches { code_completion })
    }
}

/// Custom function to convert a fill-in-the-middle (FIM) code completion response to a patch
///
/// # Notes
///
/// * LHS schema is Workspace
/// * RHS schema is Message
/// * Output schema is Patch
///
/// # Arguments
///
/// * `lhs_args` - Slice of [RecordBatch]es with the assistant FIM code completion
/// * `device` - The compute device
pub fn from_messages_to_patches(
    lhs_args: &[RecordBatch],
    code_completion: &CodeCompletionType,
    _device: &Device,
) -> Result<RecordBatch> {
    // Wrap the lhs and rhs into tables
    let lhs_table = Subject::get_builder()
        .with_record_batches(lhs_args.to_vec())?
        .with_name("from_messages_to_patches Code Completion")
        .build()?;

    // Get the content
    let content_str = lhs_table.get_column_as_vec_nonprimitive::<String>("content")?;
    let content_str = content_str.last().ok_or(anyhow!("Missing code completion content."))?;

    // Parse the content
    let (filenames, diffs, operators) = match code_completion {
        CodeCompletionType::FIM => {
            let diffs = parse_fill_in_the_middle_output(content_str);            
            let filename = vec![diffs.first().unwrap().filename.clone(), diffs.last().unwrap().filename.clone()];
            let diff = vec![diffs.first().unwrap().diff.clone(), diffs.last().unwrap().diff.clone()];
            let operator = vec![PatchOperator::Update.to_string(), PatchOperator::Update.to_string()];
            (filename, diff, operator)
        },
        CodeCompletionType::SRI => {
            let diff = parse_search_and_replace_output(content_str);
            (vec![diff.filename], vec![diff.diff], vec![PatchOperator::Update.to_string()])
        }
    };

    // Create the patch batch
    create_workspace_patch_batch(filenames, diffs, operators)
}

#[cfg(test)]
mod tests {
    use crate::device;
    use phymes_schemas::create_chat_record_batch;

    use super::*;

    #[test]
    fn test_from_messages_to_patches_sri() -> Result<()> {
        // Create the mock repository
        let role = [
            "user",
            "assistant",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let content = [
            r#"<|repo_name|>library-system
<|file_sep|>/src/library.py
class Book:
    def __init__(self, title, author, isbn, copies):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.copies = copies

    def __str__(self):
        return f"Title: {self.title}, Author: {self.author}, ISBN: {self.isbn}, Copies: {self.copies}"

class Library:
    def __init__(self):
        self.books = []

    def add_book(self, title, author, isbn, copies):
        book = Book(title, author, isbn, copies)
        self.books.append(book)

    def find_book(self, isbn):
        for book in self.books:
            if book.isbn == isbn:
                return book
        return None

    def list_books(self):
        return self.books

<|file_sep|>/src/student.py
class Student:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.borrowed_books = []

    def borrow_book(self, book, library):
        if book and book.copies > 0:
            self.borrowed_books.append(book)
            book.copies -= 1
            return True
        return False

    def return_book(self, book, library):
        if book in self.borrowed_books:
            self.borrowed_books.remove(book)
            book.copies += 1
            return True
        return False

<|file_sep|>/src/main.py
from library import Library
from student import Student

def main():
    # Set up the library with some books
    library = Library()
    library.add_book("The Great Gatsby", "F. Scott Fitzgerald", "1234567890", 3)
    library.add_book("To Kill a Mockingbird", "Harper Lee", "1234567891", 2)
    
    # Set up a student
    student = Student("Alice", "S1")
    
    # Student borrows a book
    /* MIDDLE CODE TO COMPLETE */
    if student.borrow_book(book, library):
        print(f"{student.name} borrowed {book.title}")
    else:
        print(f"{student.name} could not borrow {book.title}")
        
    # Student returns a book
    if student.return_book(book, library):
        print(f"{student.name} returned {book.title}")
    else:
        print(f"{student.name} could not return {book.title}")
    
    # List all books in the library
    print("All books in the library:")
    for book in library.list_books():
        print(book)

if __name__ == "__main__":
    main()"#,
            "```\n/src/main.py\n<<<<<<< SEARCH\n    /* MIDDLE CODE TO COMPLETE */\n=======\n    book = library.find_book(\"1234567890\")\n>>>>>>> REPLACE\n```",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let timestamp = [0, 1]
        .into_iter()
        .map(|s| s as i64)
        .collect::<Vec<_>>();
        let batch = create_chat_record_batch(role, content, timestamp)?;

        // Make the device
        let device = device(false)?;

        let result = from_messages_to_patches(&[batch], &CodeCompletionType::SRI, &device)?;
        let result_table = Subject::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let cols = result_table.get_column_as_vec_str("filename");
        assert_eq!(cols, ["/src/main.py"]);
        let cols = result_table.get_column_as_vec_str("diff");
        assert_eq!(cols, ["<<<<<<< SEARCH\n    /* MIDDLE CODE TO COMPLETE */\n=======\n    book = library.find_book(\"1234567890\")\n>>>>>>> REPLACE\n"]);
        let cols = result_table.get_column_as_vec_str("operator");
        assert_eq!(cols, ["Update"]);

        Ok(())
    }

    #[test]
    fn test_from_messages_to_patches_fim() -> Result<()> {
        // Create the mock repository
        let role = [
            "user",
            "assistant",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let content = [
            r#"<|repo_name|>library-system
<|file_sep|>/src/library.py
class Book:
    def __init__(self, title, author, isbn, copies):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.copies = copies

    def __str__(self):
        return f"Title: {self.title}, Author: {self.author}, ISBN: {self.isbn}, Copies: {self.copies}"

class Library:
    def __init__(self):
        self.books = []

    def add_book(self, title, author, isbn, copies):
        book = Book(title, author, isbn, copies)
        self.books.append(book)

    def find_book(self, isbn):
        for book in self.books:
            if book.isbn == isbn:
                return book
        return None

    def list_books(self):
        return self.books

<|file_sep|>/src/student.py
class Student:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.borrowed_books = []

    def borrow_book(self, book, library):
        if book and book.copies > 0:
            self.borrowed_books.append(book)
            book.copies -= 1
            return True
        return False

    def return_book(self, book, library):
        if book in self.borrowed_books:
            self.borrowed_books.remove(book)
            book.copies += 1
            return True
        return False

<|file_sep|>/src/main.py
<|fim_prefix|>from library import Library
from student import Student

def main():
    # Set up the library with some books
    library = Library()
    library.add_book("The Great Gatsby", "F. Scott Fitzgerald", "1234567890", 3)
    library.add_book("To Kill a Mockingbird", "Harper Lee", "1234567891", 2)
    
    # Set up a student
    student = Student("Alice", "S1")
    
    # Student borrows a book<|fim_suffix|>
    if student.borrow_book(book, library):
        print(f"{student.name} borrowed {book.title}")
    else:
        print(f"{student.name} could not borrow {book.title}")
        
    # Student returns a book
    if student.return_book(book, library):
        print(f"{student.name} returned {book.title}")
    else:
        print(f"{student.name} could not return {book.title}")
    
    # List all books in the library
    print("All books in the library:")
    for book in library.list_books():
        print(book)

if __name__ == "__main__":
    main()<|fim_middle|>"#,
            "/src/main.py\n```python\n    book = library.find_book(\"1234567890\")\n```",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
        let timestamp = [0, 1]
        .into_iter()
        .map(|s| s as i64)
        .collect::<Vec<_>>();
        let batch = create_chat_record_batch(role, content, timestamp)?;

        // Make the device
        let device = device(false)?;

        let result = from_messages_to_patches(&[batch], &CodeCompletionType::FIM, &device)?;
        let result_table = Subject::get_builder()
            .with_record_batches(vec![result])?
            .with_name("")
            .build()?;

        let cols = result_table.get_column_as_vec_str("filename");
        assert_eq!(cols, ["/src/main.py", "/src/main.py"]);
        let cols = result_table.get_column_as_vec_str("diff");
        assert_eq!(cols, [
            "<<<<<<< SEARCH\n<|fim_prefix|>=======\n>>>>>>> REPLACE\n",
            "<<<<<<< SEARCH\n<|fim_suffix|>=======\n\n    book = library.find_book(\"1234567890\")\n>>>>>>> REPLACE",
            ]);
        let cols = result_table.get_column_as_vec_str("operator");
        assert_eq!(cols, ["Update", "Update"]);

        Ok(())
    }
}
