import { Component, signal, computed, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

interface SearchResult {
  id: number;
  title: string;
  author: string;
  subtitle: string;
  text: string;
  date: string;
  imageUrl: string;
  link: string;
  isRelevant?: boolean; /* Track if it is relevant/irrelevant locally */
}

@Component({
  selector: 'app-root',
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  protected readonly title = signal('Frontend');
  protected isLoading = signal<boolean>(false);
  
  // Error state for UI
  showError = signal<boolean>(false);
  errorMessage = signal<string>('');
  current_query = signal<string>('');
  
  // Model scores from backend feedback
  modelScores = signal<Record<string, number>>({});
  
  // Dynamically sorted algorithms based on backend scores (highest to lowest)
  sortedAlgorithms = computed(() => {
    const scores = this.modelScores();
    return [...this.algorithms].sort((a, b) => {
      const scoreA = scores[a.id] !== undefined ? scores[a.id] : -1;
      const scoreB = scores[b.id] !== undefined ? scores[b.id] : -1;
      return scoreB - scoreA;
    });
  });
  
  // titleText = signal('Search for a Document');
  query_result = signal<any>({});
  
  selectedAlgorithm: string = 'rm3';
  algorithms = [
    { id: 'rm3', name: 'RM3' },
    { id: 'bm25', name: 'BM25' },
    { id: 'tfidf', name: 'TF-IDF' },
    { id: 'cosine', name: 'Cosine Similarity' },
    { id: 'gram', name: 'N-Gram (3-max)' },
    { id: 'elmo', name: 'ELMO (always searches in text)' }
  ];

  selectedField: string = 'text';
  searchFields = [
    { id: 'text', name: 'Text (Default)' },
    { id: 'title', name: 'Title' },
    { id: 'subtitle', name: 'Subtitle' },
    // { id: 'author', name: 'Author' }
  ];

  // Suggestions state
  showSuggestions: boolean = false;
  suggestions = signal<string[]>([]);
  private debounceTimer: any;

  
  onSuggestionSelect(suggestion: string, searchInput: HTMLInputElement): void {
    // Fill the input box only
    searchInput.value = suggestion;
    
    // Hide the suggestions dropdown
    this.showSuggestions = false;
  }
  
  results: SearchResult[] = [
    {
      id: 1,
      title: 'Understanding Artificial Intelligence',
      author: 'By Dr. John Smith',
      subtitle: 'A comprehensive guide to modern AI technologies',
      text: 'Artificial Intelligence has revolutionized the way we approach problem-solving in the modern world. From machine learning algorithms to deep neural networks, AI technologies continue to evolve and improve our daily lives. This document explores the fundamental concepts...',
      date: 'Published: March 15, 2024',
      imageUrl: 'https://via.placeholder.com/200x250/667eea/ffffff?text=Document',
      link: ''
    }
  ];

  ngOnInit(): void {
    this.recommend();
    this.updatescore();
  }

  // ===============================
  // ===========Functions===========
  // ===============================

  async submitRelevance(docId: number, isRelevant: boolean): Promise<void> {
    const result = this.results.find(r => r.id === docId);
    if (result) {
      result.isRelevant = isRelevant;
    }

    const allResultIds = this.results.map(r => r.id);

    try {
        const response = await fetch('http://localhost:5000/Feedback', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            docid: docId,
            relevant:isRelevant,
            query: this.current_query(),
            model: this.selectedAlgorithm,
            retrived: allResultIds
          })
        });
        
        if (!response.ok) {
          throw new Error('Failed to connect to server');
        }
        const data = await response.json();
        
        // Update the model scores with real-time feedback data
        this.modelScores.set(data);
        
      } catch (err: any) {
        console.error('Error:', err);
      }

  }

  onInputChange(value: string): void {
    if (!value.trim()) {
      this.showSuggestions = false;
      this.suggestions.set([]);
      return;
    }

    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(async () => {
      try {
        const response = await fetch('http://localhost:5000/suggest', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            query: value,
          })
        });
        
        if (!response.ok) {
          throw new Error('Failed to connect to server');
        }
        const data = await response.json();
        
        this.suggestions.set(data);
        this.showSuggestions = true;
        
      } catch (err: any) {
        console.error('Error:', err);
      }
    }, 300);
  }
  
  async updatescore(): Promise<void>{
    const allResultIds = this.results.map(r => r.id);

    try {
        const response = await fetch('http://localhost:5000/updatescore', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({})
        });
        
        if (!response.ok) {
          throw new Error('Failed to connect to server');
        }
        const data = await response.json();
        
        // Update the model scores with real-time feedback data
        this.modelScores.set(data);
        
      } catch (err: any) {
        console.error('Error:', err);
      }

  }

  async recommend(): Promise<void>{
    try {
      const response = await fetch('http://localhost:5000/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
      });
      
      if (!response.ok) {
        throw new Error('Failed to connect to server');
      }
      const data = await response.json();
      
      this.results = Object.values(data).map((item: any, index: number) => ({
      id: item.id,
      title: item.title,
      author: item.Author,
      subtitle: item.SubTitle,
      text: item.text,
      date: item.date,
      imageUrl: item.image,
      link: item.link
    }));
      
    } catch (err: any) {
      console.error('Error:', err);
    }
  }

  async onSearch(query:string): Promise<void>{
    this.showSuggestions = false;

    // Reset error state on new search
    this.showError.set(false);
    
    if (query.trim().length === 0) {
      this.errorMessage.set('Please enter a search query before searching.');
      this.showError.set(true);
      return;
    }

    const wordCount = query.trim().split(/\s+/).length;
    if (wordCount > 3 && this.selectedAlgorithm === "gram") {
      this.errorMessage.set('Please enter between 1-3 words search for N-Gram.');
      this.showError.set(true);
      return;
    }
    
    this.isLoading.set(true);
    this.current_query.set(query)
    try {
      const response = await fetch('http://localhost:5000/result', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: query,
          algorithm: this.selectedAlgorithm,
          field: this.selectedField
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to connect to server');
      }
      const data = await response.json();
      
      this.results = Object.values(data).map((item: any, index: number) => ({
      id: item.id,
      title: item.title,
      author: item.Author,
      subtitle: item.SubTitle,
      text: item.text,
      date: item.date,
      imageUrl: item.image,
      link: item.link
    }));
      
      this.query_result.set(data);
      this.showError.set(false); // Hide error on successful response
      
    } catch (err: any) {
      console.error('Error:', err);
      this.errorMessage.set(err.message || 'An error occurred while searching.');
      this.showError.set(true);
    } finally {
      this.isLoading.set(false);
    }
  }

}
