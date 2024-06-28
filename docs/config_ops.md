
## Configuration modifying primitives

#### `bind_config(proc, var_cursor, config, field)`
Extracts a control-value expression and write it into some designated field of a config.
```
args:
    var_cursor  - cursor or pattern pointing at the expression to be bound
    config      - config object to be written into
    field       - (string) the field of `config` to be written to

rewrite:
    Let s[ e ] mean a statement with control expression e occurring within it. Then,
    s[ e ]
      -->
    config.field = e
    s[ config.field ]
```

#### `delete_config(proc, stmt_cursor)`
Delete a statement that writes to some config.field.
```
args:
    stmt_cursor - cursor or pattern pointing at the statement to
                  be deleted

rewrite:
    s1
    config.field = _
    s3
      -->
    s1
    s3
```

#### `write_config(proc, gap_cursor, config, field, rhs)`
Inserts a statement that writes a desired value to some config.field.
```
args:
    gap_cursor  - cursor pointing to where the new write statement should be inserted
    config      - config object to be written into
    field       - (string) the field of `config` to be written to
    rhs         - (string) the expression to write into the field

rewrite:
    s1
    s3
      -->
    s1
    config.field = new_expr
    s3
```
