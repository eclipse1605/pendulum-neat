                      
import os
import tokenize


def remove_comments_from_file(filepath):
    """Read a Python file, strip out comment tokens, and overwrite the file."""
                                                    
    with open(filepath, 'rb') as f:
        tokens = tokenize.tokenize(f.readline)
        result_tokens = []

        for tok in tokens:
                                                     
            if tok.type in (tokenize.COMMENT, tokenize.ENCODING):
                continue
            result_tokens.append(tok)

                                          
    new_source = tokenize.untokenize(result_tokens)

                             
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_source)


def main():
    for root, dirs, files in os.walk('.'):
                                          
        dirs[:] = [d for d in dirs if d not in ('.venv', '__pycache__')]

        for filename in files:
            if not filename.endswith('.py'):
                continue

            filepath = os.path.join(root, filename)
            try:
                remove_comments_from_file(filepath)
                print(f"Processed: {filepath}")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")


if __name__ == '__main__':
    main()
