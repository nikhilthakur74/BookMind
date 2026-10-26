import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CitationsView } from '../components/CitationsView';
import type { Citation } from '../types/api';

describe('CitationsView', () => {
  const mockCitations: Citation[] = [
    {
      chunk_id: 'book_abc_c0001',
      page_start: 3,
      page_end: 4,
      quoted_snippet: 'Distributed consensus requires quorum agreement across nodes.',
      chapter_title: 'Chapter 2: Fault Tolerance',
      section_title: 'Section 2.1: Partitions',
      confidence: 'high',
    },
    {
      chunk_id: 'book_abc_c0002',
      page_start: 7,
      page_end: 7,
      quoted_snippet: 'Byzantine faults assume arbitrary node malfunction.',
      confidence: 'medium',
    },
  ];

  it('renders citation count and badges', () => {
    render(<CitationsView citations={mockCitations} />);

    expect(screen.getByText(/2 Verified Source Passages/)).toBeDefined();
    expect(screen.getByText('Pages 3–4')).toBeDefined();
    expect(screen.getByText('Page 7')).toBeDefined();
    expect(screen.getByText('book_abc_c0001')).toBeDefined();
  });

  it('expands snippet on click', () => {
    render(<CitationsView citations={mockCitations} />);

    // Initially snippet not expanded
    expect(screen.queryByText(/Distributed consensus requires quorum/)).toBeNull();

    // Click on first citation header
    const citeButton = screen.getByText('Pages 3–4');
    fireEvent.click(citeButton);

    // Now snippet is visible
    expect(screen.getByText(/"Distributed consensus requires quorum agreement across nodes."/)).toBeDefined();
    expect(screen.getByText('Section 2.1: Partitions')).toBeDefined();
  });

  it('supports Expand All and Collapse All toggle', () => {
    render(<CitationsView citations={mockCitations} />);

    const expandAllBtn = screen.getByText('Expand All');
    fireEvent.click(expandAllBtn);

    // Both snippets visible
    expect(screen.getByText(/"Distributed consensus requires quorum agreement across nodes."/)).toBeDefined();
    expect(screen.getByText(/"Byzantine faults assume arbitrary node malfunction."/)).toBeDefined();

    // Collapse All
    const collapseBtn = screen.getByText('Collapse All');
    fireEvent.click(collapseBtn);

    expect(screen.queryByText(/"Distributed consensus requires quorum agreement across nodes."/)).toBeNull();
  });
});
