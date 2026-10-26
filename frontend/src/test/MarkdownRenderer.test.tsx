import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MarkdownRenderer } from '../components/MarkdownRenderer';

describe('MarkdownRenderer', () => {
  it('renders plain text and headings correctly', () => {
    const markdown = `# Title 1\n## Section 2\n### Subsection 3\nThis is regular body text.`;
    render(<MarkdownRenderer content={markdown} />);

    expect(screen.getByText('Title 1')).toBeDefined();
    expect(screen.getByText('Section 2')).toBeDefined();
    expect(screen.getByText('Subsection 3')).toBeDefined();
    expect(screen.getByText(/This is regular body text/)).toBeDefined();
  });

  it('renders bold and italic formatting', () => {
    const markdown = `The **essential crux** of the book is *grounded evidence*.`;
    const { container } = render(<MarkdownRenderer content={markdown} />);

    const strongElement = container.querySelector('strong');
    expect(strongElement).toBeDefined();
    expect(strongElement?.textContent).toBe('essential crux');

    const emElement = container.querySelector('em');
    expect(emElement).toBeDefined();
    expect(emElement?.textContent).toBe('grounded evidence');
  });

  it('renders unordered and ordered list items', () => {
    const markdown = `- First item\n- Second item\n1. Step one\n2. Step two`;
    render(<MarkdownRenderer content={markdown} />);

    expect(screen.getByText('First item')).toBeDefined();
    expect(screen.getByText('Second item')).toBeDefined();
    expect(screen.getByText('Step one')).toBeDefined();
    expect(screen.getByText('Step two')).toBeDefined();
  });

  it('renders blockquotes with citation styles', () => {
    const markdown = `> Author states that distributed consensus requires majority quorum.`;
    const { container } = render(<MarkdownRenderer content={markdown} />);

    const blockquote = container.querySelector('blockquote');
    expect(blockquote).toBeDefined();
    expect(blockquote?.textContent).toContain('distributed consensus requires majority quorum');
  });

  it('renders code blocks safely', () => {
    const markdown = "```python\ndef solve():\n    return True\n```";
    const { container } = render(<MarkdownRenderer content={markdown} />);

    const pre = container.querySelector('pre');
    expect(pre).toBeDefined();
    expect(pre?.textContent).toContain('def solve()');
  });
});
