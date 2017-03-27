{ open Parser }

rule token =
  parse [ ' ' '\t' '\r' '\n' ] { token lexbuf }
    |   '+'                    { PLUS }
    |   '-'                    { MINUS }
    |   '*'                    { TIMES }
    |   '/'                    { DIVIDE }
    |   '='                    { ASSIGN }
    |   ','                    { SEQUENCE }
    |   ['0'-'9']+ as lit      { LITERAL(int_of_string lit) }
    |   '$'['a'-'z'] as lit    { VARIABLE(int_of_char lit.[1] - 97) }
    |   eof                    { EOF }
