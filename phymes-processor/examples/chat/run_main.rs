#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use clap::Parser;
use phymes_ml::CandleChatConfig;
use phymes_processor::bench_chat_processor::bench_chat_processor;
use phymes_subject::SubjectTrait;

pub async fn run_main() -> Result<()> {
    // DM, todo!(): move to phymes-ml
    // println!(
    //     "avx: {}, neon: {}, simd128: {}, f16c: {}",
    //     candle_core::utils::with_avx(),
    //     candle_core::utils::with_neon(),
    //     candle_core::utils::with_simd128(),
    //     candle_core::utils::with_f16c()
    // );

    // Chat processor config
    let config = CandleChatConfig::parse();

    // Run the chat processor
    let message_history = bench_chat_processor(
        None,
        &config,
        "What are the four molecules that compose DNA?",
//         r#"<|repo_name|>library-system
// <|file_sep|>library.py
// class Book:
//     def __init__(self, title, author, isbn, copies):
//         self.title = title
//         self.author = author
//         self.isbn = isbn
//         self.copies = copies

//     def __str__(self):
//         return f"Title: {self.title}, Author: {self.author}, ISBN: {self.isbn}, Copies: {self.copies}"

// class Library:
//     def __init__(self):
//         self.books = []

//     def add_book(self, title, author, isbn, copies):
//         book = Book(title, author, isbn, copies)
//         self.books.append(book)

//     def find_book(self, isbn):
//         for book in self.books:
//             if book.isbn == isbn:
//                 return book
//         return None

//     def list_books(self):
//         return self.books

// <|file_sep|>student.py
// class Student:
//     def __init__(self, name, id):
//         self.name = name
//         self.id = id
//         self.borrowed_books = []

//     def borrow_book(self, book, library):
//         if book and book.copies > 0:
//             self.borrowed_books.append(book)
//             book.copies -= 1
//             return True
//         return False

//     def return_book(self, book, library):
//         if book in self.borrowed_books:
//             self.borrowed_books.remove(book)
//             book.copies += 1
//             return True
//         return False

// <|file_sep|>main.py
// <|fim_prefix|>from library import Library
// from student import Student

// def main():
//     # Set up the library with some books
//     library = Library()
//     library.add_book("The Great Gatsby", "F. Scott Fitzgerald", "1234567890", 3)
//     library.add_book("To Kill a Mockingbird", "Harper Lee", "1234567891", 2)
    
//     # Set up a student
//     student = Student("Alice", "S1")
    
//     # Student borrows a book<|fim_suffix|>
//     if student.borrow_book(book, library):
//         print(f"{student.name} borrowed {book.title}")
//     else:
//         print(f"{student.name} could not borrow {book.title}")
        
//     # Student returns a book
//     if student.return_book(book, library):
//         print(f"{student.name} returned {book.title}")
//     else:
//         print(f"{student.name} could not return {book.title}")
    
//     # List all books in the library
//     print("All books in the library:")
//     for book in library.list_books():
//         print(book)

// if __name__ == "__main__":
//     main()<|fim_middle|>"#,
        "chat_processor",
    )
    .await?;
    let json_data = message_history.to_json_object()?;
    for row in json_data {
        if row["role"] != "system" {
            println!("{} @ {}: {}", row["role"], row["timestamp"], row["content"])
        }
    }

    Ok(())
}
