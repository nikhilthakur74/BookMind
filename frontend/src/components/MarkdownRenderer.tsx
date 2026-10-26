import React from 'react';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

type BlockType =
  | { type: 'code'; lang: string; code: string }
  | { type: 'h1'; text: string }
  | { type: 'h2'; text: string }
  | { type: 'h3'; text: string }
  | { type: 'blockquote'; lines: string[] }
  | { type: 'ul'; items: string[] }
  | { type: 'ol'; items: string[] }
  | { type: 'p'; text: string };

function parseBlocks(markdown: string): BlockType[] {
  const lines = markdown.split(/\r?\n/);
  const blocks: BlockType[] = [];

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // Empty lines
    if (!trimmed) {
      i++;
      continue;
    }

    // 1. Code Block
    if (trimmed.startsWith('```')) {
      const lang = trimmed.replace('```', '').trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      if (i < lines.length) i++; // consume closing ```
      blocks.push({ type: 'code', lang, code: codeLines.join('\n') });
      continue;
    }

    // 2. Headings
    if (trimmed.startsWith('# ')) {
      blocks.push({ type: 'h1', text: trimmed.replace(/^#\s+/, '') });
      i++;
      continue;
    }
    if (trimmed.startsWith('## ')) {
      blocks.push({ type: 'h2', text: trimmed.replace(/^##\s+/, '') });
      i++;
      continue;
    }
    if (trimmed.startsWith('### ')) {
      blocks.push({ type: 'h3', text: trimmed.replace(/^###\s+/, '') });
      i++;
      continue;
    }

    // 3. Blockquote
    if (trimmed.startsWith('>')) {
      const bqLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        bqLines.push(lines[i].trim().replace(/^>\s?/, ''));
        i++;
      }
      blocks.push({ type: 'blockquote', lines: bqLines });
      continue;
    }

    // 4. Unordered List
    if (/^[-*]\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ''));
        i++;
      }
      blocks.push({ type: 'ul', items });
      continue;
    }

    // 5. Ordered List
    if (/^\d+\.\s+/.test(trimmed)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''));
        i++;
      }
      blocks.push({ type: 'ol', items });
      continue;
    }

    // 6. Regular Paragraph
    const pLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trim().startsWith('```') &&
      !lines[i].trim().startsWith('#') &&
      !lines[i].trim().startsWith('>') &&
      !/^[-*]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim())
    ) {
      pLines.push(lines[i]);
      i++;
    }
    blocks.push({ type: 'p', text: pLines.join(' ') });
  }

  return blocks;
}

/**
 * Lightweight, zero-dependency Markdown renderer tailored for reading and chat.
 * Safely formats code blocks, headings, bold, italics, quotes, lists, and links.
 */
export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  if (!content) return null;

  const blocks = parseBlocks(content);

  return (
    <div className={`space-y-3.5 leading-relaxed text-[#2D2B28] ${className}`}>
      {blocks.map((block, idx) => {
        switch (block.type) {
          case 'code':
            return (
              <div
                key={idx}
                className="my-3 overflow-hidden rounded-xl border border-[#E5E0D8] bg-[#22211F] text-[#F5F2EB]"
              >
                {block.lang && (
                  <div className="border-b border-[#383531] bg-[#1A1918] px-3.5 py-1.5 font-mono-code text-[11px] text-[#A6A095]">
                    {block.lang}
                  </div>
                )}
                <pre className="overflow-x-auto p-3.5 font-mono-code text-xs leading-relaxed">
                  <code>{block.code}</code>
                </pre>
              </div>
            );

          case 'blockquote':
            return (
              <blockquote
                key={idx}
                className="my-3 border-l-2 border-[#C86432] bg-[#FAF6F0] py-2 pl-4 pr-3 font-serif-claude text-[15px] italic text-[#4D4942]"
              >
                {block.lines.map((l, qIdx) => (
                  <p key={qIdx}>{renderInlineMarkdown(l)}</p>
                ))}
              </blockquote>
            );

          case 'h1':
            return (
              <h1
                key={idx}
                className="font-serif-claude text-2xl font-normal text-[#22211F] pt-3 pb-1"
              >
                {renderInlineMarkdown(block.text)}
              </h1>
            );

          case 'h2':
            return (
              <h2
                key={idx}
                className="font-serif-claude text-xl font-medium text-[#22211F] pt-3 pb-1 border-b border-[#ECE7DE]"
              >
                {renderInlineMarkdown(block.text)}
              </h2>
            );

          case 'h3':
            return (
              <h3 key={idx} className="font-serif-claude text-lg font-medium text-[#22211F] pt-2">
                {renderInlineMarkdown(block.text)}
              </h3>
            );

          case 'ul':
            return (
              <ul
                key={idx}
                className="my-2 list-outside list-disc space-y-1.5 pl-5 text-[14px] text-[#3D3A36]"
              >
                {block.items.map((item, iIdx) => (
                  <li key={iIdx}>{renderInlineMarkdown(item)}</li>
                ))}
              </ul>
            );

          case 'ol':
            return (
              <ol
                key={idx}
                className="my-2 list-outside list-decimal space-y-1.5 pl-5 text-[14px] text-[#3D3A36]"
              >
                {block.items.map((item, iIdx) => (
                  <li key={iIdx}>{renderInlineMarkdown(item)}</li>
                ))}
              </ol>
            );

          case 'p':
          default:
            return (
              <p key={idx} className="text-[14px] text-[#2D2B28] leading-relaxed">
                {renderInlineMarkdown(block.text)}
              </p>
            );
        }
      })}
    </div>
  );
};

/**
 * Safely parses inline markdown syntax (bold, italic, code, links).
 */
function renderInlineMarkdown(text: string): React.ReactNode {
  if (!text) return null;

  // Split tokens by code, bold, italic, and markdown links
  const regex = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g;
  const parts = text.split(regex);

  return parts.map((part, index) => {
    if (!part) return null;

    // Inline code: `code`
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return (
        <code
          key={index}
          className="rounded border border-[#E0D9CF] bg-[#F4EFE6] px-1.5 py-0.5 font-mono-code text-[12px] text-[#7C3A1E]"
        >
          {part.slice(1, -1)}
        </code>
      );
    }

    // Bold: **text**
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return (
        <strong key={index} className="font-semibold text-[#1C1A18]">
          {part.slice(2, -2)}
        </strong>
      );
    }

    // Italic: *text*
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
      return (
        <em key={index} className="italic text-[#3F3C37]">
          {part.slice(1, -1)}
        </em>
      );
    }

    // Markdown link: [text](url)
    const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (linkMatch) {
      const [, label, url] = linkMatch;
      return (
        <a
          key={index}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-[#C86432] underline underline-offset-2 hover:text-[#9A421A] transition-colors"
        >
          {label}
        </a>
      );
    }

    // Plain text
    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}
