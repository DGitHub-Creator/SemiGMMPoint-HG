# Project working agreements

- Commit every major project modification to the local Git repository after
  proportionate validation. Keep each commit scoped to the modification and do
  not include unrelated pre-existing working-tree changes.
- When GitHub or another external service requires the server proxy, export:

  ```bash
  export http_proxy=192.168.217.99:7893
  export https_proxy=192.168.217.99:7893
  ```

- Preserve existing experimental artifacts and uncommitted adaptation work.
  Never use destructive cleanup commands unless the user explicitly requests
  them.
