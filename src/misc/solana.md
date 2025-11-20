# Solana

A comprehensive guide to Solana blockchain, its architecture, and developing high-performance decentralized applications.

## Table of Contents

1. [Solana Overview](#solana-overview)
2. [Core Concepts and Architecture](#core-concepts-and-architecture)
3. [Account Model](#account-model)
4. [Programming Model](#programming-model)
5. [Development with Rust](#development-with-rust)
6. [Anchor Framework](#anchor-framework)
7. [SPL Token Program](#spl-token-program)
8. [Client Development](#client-development)
9. [CLI Tools](#cli-tools)
10. [Development Tools and Resources](#development-tools-and-resources)

---

## Solana Overview

**Solana** is a high-performance blockchain designed for decentralized applications and crypto-currencies. It aims to solve the blockchain trilemma by achieving scalability without sacrificing security or decentralization.

### Key Features

- **High Throughput**: 65,000+ theoretical transactions per second (TPS)
- **Low Latency**: ~400ms block time
- **Low Fees**: Average transaction cost < $0.001
- **Scalability**: Scales with hardware improvements
- **Proof of History**: Novel timekeeping mechanism
- **Rust-based**: Programs written in Rust for safety and performance

### Performance Characteristics

| Metric | Solana | Ethereum | Bitcoin |
|--------|--------|----------|---------|
| TPS | 65,000+ | 15-30 | 7 |
| Block Time | ~400ms | ~12s | ~10min |
| Finality | ~13s | 13+ min | 60+ min |
| Avg Fee | < $0.001 | $1-50+ | $1-20+ |
| Consensus | PoH + PoS | PoS | PoW |

### Why Solana?

1. **Speed**: Fast block times enable real-time applications
2. **Cost**: Low fees make micro-transactions viable
3. **Composability**: Parallel execution enables complex DeFi protocols
4. **Developer Experience**: Modern tooling and frameworks
5. **Growing Ecosystem**: Active developer community and projects

---

## Core Concepts and Architecture

Solana's performance comes from eight key innovations working together:

### 1. Proof of History (PoH)

**Proof of History** is a cryptographic clock that allows nodes to agree on time without communication overhead.

#### How PoH Works

PoH creates a historical record proving an event occurred at a specific moment in time:

```
Input → SHA-256 → Output
Output → SHA-256 → Output
Output → SHA-256 → Output
...

Each hash depends on the previous, creating verifiable passage of time
```

**Example:**
```
Hash 1: 0x5d2a...
Hash 2: 0x9f3b... (includes Hash 1)
Hash 3: 0x1e4c... (includes Hash 2)
```

Since each hash requires the previous one, you can't compute Hash 3 without first computing Hash 1 and Hash 2. This proves sequence and time.

**Benefits:**
- Nodes don't need to wait for messages about time
- Reduces consensus overhead
- Enables parallel transaction processing
- Predictable block production

### 2. Tower BFT

**Tower BFT**: Solana's PBFT-like consensus algorithm optimized with PoH

- Uses PoH as a cryptographic clock
- Validators vote on PoH hashes
- Votes are weighted by stake
- Timeout-based finality (no communication rounds)

**Finality:**
- Optimistic confirmation: < 1 second
- Full finality: ~13 seconds (32 confirmed blocks)

### 3. Turbine

**Turbine**: Block propagation protocol inspired by BitTorrent

```
Leader
  ├─→ Validator Layer 1 (3 nodes)
  │    ├─→ Validator Layer 2 (9 nodes)
  │    │    └─→ Validator Layer 3 (27 nodes)
  │    ├─→ Validator Layer 2 (9 nodes)
  │    └─→ Validator Layer 2 (9 nodes)
  └─→ ...
```

- Breaks blocks into packets
- Distributes via tree structure
- Reduces bandwidth requirements
- Enables fast block propagation

### 4. Gulf Stream

**Gulf Stream**: Mempool-less transaction forwarding

- Transactions sent to upcoming leaders before their slot
- Leaders execute transactions before becoming leader
- Reduces confirmation time
- Enables 50,000+ TPS

### 5. Sealevel

**Sealevel**: Parallel smart contract runtime

- First parallel execution environment for smart contracts
- Executes thousands of contracts in parallel
- Uses account locking for concurrency control
- Scales with number of CPU cores

**Example:**
```
Transaction 1: Account A → Account B
Transaction 2: Account C → Account D
(Can execute in parallel - no shared state)

Transaction 3: Account A → Account E
Transaction 1: Account A → Account B
(Cannot execute in parallel - both use Account A)
```

### 6. Pipelining

**Pipelining**: Transaction processing optimization

Stages:
```
1. Data Fetch (Kernel) → 2. Signature Verify (GPU) →
3. Banking (CPU) → 4. Write (Kernel)
```

- Different stages run on different hardware
- Continuous flow like assembly line
- Maximizes hardware utilization

### 7. Cloudbreak

**Cloudbreak**: Horizontally-scaled accounts database

- Account state stored in memory-mapped files
- Simultaneous reads across SSDs
- Scales with disk count
- Enables millions of accounts

### 8. Archivers

**Archivers**: Distributed ledger storage

- Offload history from validators
- Proof-of-replication for data integrity
- Reduces validator storage burden
- Enables long-term data availability

---

## Account Model

Unlike Ethereum's account-based model, Solana uses an **account model** where everything is an account.

### Account Structure

```rust
pub struct Account {
    pub lamports: u64,        // Account balance in lamports
    pub data: Vec<u8>,        // Stored data
    pub owner: Pubkey,        // Program that owns this account
    pub executable: bool,     // Is this account a program?
    pub rent_epoch: Epoch,    // Next epoch to collect rent
}
```

### Account Types

#### 1. Program Accounts (Executable)
- Contain executable code
- `executable: true`
- Immutable after deployment
- Owned by BPF Loader

#### 2. Data Accounts
- Store program state
- `executable: false`
- Can be modified by owning program
- Created by programs

#### 3. Native Accounts
- System accounts (wallet addresses)
- Owned by System Program
- Store SOL balance

### Account Ownership

**Key Principle**: Only the owner program can modify an account's data

```
User Wallet (System Program owned)
  → Can only debit lamports with signature

Data Account (Custom Program owned)
  → Only Custom Program can modify data
  → User must call Custom Program to update
```

### Rent

Accounts must maintain minimum balance for **rent exemption**.

```rust
// Calculate rent-exempt balance
let rent = Rent::get()?;
let min_balance = rent.minimum_balance(account_data_len);

// Accounts with balance >= min_balance are rent-exempt
// Accounts below threshold lose lamports each epoch
```

**Rent Calculation:**
```
Rent = account_size_in_bytes * price_per_byte_epoch * epochs
```

**Best Practice**: Always make accounts rent-exempt

### System Program

The **System Program** manages:
- Creating new accounts
- Allocating account space
- Transferring lamports
- Assigning account ownership

```rust
// Create a new account
system_instruction::create_account(
    &payer,           // Who pays for account creation
    &new_account,     // New account address
    lamports,         // Initial balance (rent-exempt)
    space,            // Data size in bytes
    &owner_program,   // Program that will own account
)
```

---

## Programming Model

Solana programs (smart contracts) are **stateless** - they don't store state internally. All state is stored in accounts.

### Programs vs Accounts

```
┌─────────────────┐
│  Program        │ (Executable, Immutable)
│  - Logic only   │
│  - No state     │
└─────────────────┘
        ↓
   Operates on
        ↓
┌─────────────────┐
│  Data Accounts  │ (Non-executable, Mutable)
│  - Store state  │
│  - Owned by     │
│    program      │
└─────────────────┘
```

### Instructions and Transactions

**Instruction**: Single operation to execute on a program

```rust
pub struct Instruction {
    pub program_id: Pubkey,              // Program to call
    pub accounts: Vec<AccountMeta>,      // Accounts involved
    pub data: Vec<u8>,                   // Instruction data
}
```

**Transaction**: One or more instructions executed atomically

```rust
pub struct Transaction {
    pub signatures: Vec<Signature>,      // Required signatures
    pub message: Message,                // Instructions + accounts
}
```

**Atomic Execution:**
- All instructions succeed, or all fail
- No partial state changes
- Ensures consistency

### Account Metadata

Each instruction specifies how accounts are used:

```rust
pub struct AccountMeta {
    pub pubkey: Pubkey,      // Account address
    pub is_signer: bool,     // Must sign transaction?
    pub is_writable: bool,   // Will be modified?
}
```

### Program Derived Addresses (PDAs)

**PDA**: Address derived from program ID and seeds (no private key exists)

```rust
// Find PDA
let (pda, bump_seed) = Pubkey::find_program_address(
    &[b"seeds", user_pubkey.as_ref()],
    &program_id,
);

// PDA properties:
// - Deterministic (same seeds → same address)
// - Off the ed25519 curve (no private key)
// - Only owning program can sign for PDA
```

**Use Cases:**
- Program-owned accounts
- Storing program state
- Signing transactions on behalf of program
- Deterministic account addresses

### Cross-Program Invocations (CPI)

Programs can call other programs:

```rust
invoke(
    &instruction,    // Instruction to invoke
    &accounts,       // Accounts to pass
)?;

// Or with PDA signing
invoke_signed(
    &instruction,
    &accounts,
    &[&[b"seeds", &[bump_seed]]],  // Seeds for PDA signature
)?;
```

**CPI Depth**: Maximum 4 levels deep

---

## Development with Rust

### Environment Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Solana CLI
sh -c "$(curl -sSfL https://release.solana.com/stable/install)"

# Verify installation
solana --version
rustc --version

# Set config to devnet
solana config set --url https://api.devnet.solana.com

# Create a keypair
solana-keygen new --outfile ~/.config/solana/id.json

# Get some SOL for testing
solana airdrop 2
```

### Project Structure

```bash
# Create new project
cargo new --lib my-solana-program
cd my-solana-program

# Add dependencies to Cargo.toml
[dependencies]
solana-program = "1.18"

[lib]
crate-type = ["cdylib", "lib"]
```

### Hello World Program

```rust
use solana_program::{
    account_info::AccountInfo,
    entrypoint,
    entrypoint::ProgramResult,
    msg,
    pubkey::Pubkey,
};

// Declare program entrypoint
entrypoint!(process_instruction);

// Program entrypoint function
pub fn process_instruction(
    program_id: &Pubkey,        // This program's address
    accounts: &[AccountInfo],   // Accounts passed to program
    instruction_data: &[u8],    // Instruction data
) -> ProgramResult {
    msg!("Hello, Solana!");
    Ok(())
}
```

### Counter Program Example

```rust
use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::{
    account_info::{next_account_info, AccountInfo},
    entrypoint,
    entrypoint::ProgramResult,
    msg,
    program_error::ProgramError,
    pubkey::Pubkey,
};

// Define state structure
#[derive(BorshSerialize, BorshDeserialize, Debug)]
pub struct CounterAccount {
    pub count: u32,
}

entrypoint!(process_instruction);

pub fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    // Parse instruction
    let instruction = instruction_data
        .get(0)
        .ok_or(ProgramError::InvalidInstructionData)?;

    // Get accounts
    let accounts_iter = &mut accounts.iter();
    let counter_account = next_account_info(accounts_iter)?;

    // Verify ownership
    if counter_account.owner != program_id {
        return Err(ProgramError::IncorrectProgramId);
    }

    // Process instruction
    match instruction {
        0 => increment(counter_account),
        1 => decrement(counter_account),
        _ => Err(ProgramError::InvalidInstructionData),
    }
}

fn increment(counter_account: &AccountInfo) -> ProgramResult {
    let mut counter = CounterAccount::try_from_slice(&counter_account.data.borrow())?;
    counter.count += 1;
    counter.serialize(&mut &mut counter_account.data.borrow_mut()[..])?;
    msg!("Counter incremented to: {}", counter.count);
    Ok(())
}

fn decrement(counter_account: &AccountInfo) -> ProgramResult {
    let mut counter = CounterAccount::try_from_slice(&counter_account.data.borrow())?;
    counter.count = counter.count.saturating_sub(1);
    counter.serialize(&mut &mut counter_account.data.borrow_mut()[..])?;
    msg!("Counter decremented to: {}", counter.count);
    Ok(())
}
```

### Building and Deploying

```bash
# Build program
cargo build-bpf

# Deploy to devnet
solana program deploy target/deploy/my_solana_program.so

# Output:
# Program Id: 7X8y9Z3a...
```

### Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use solana_program::clock::Epoch;
    use std::mem;

    #[test]
    fn test_increment() {
        let program_id = Pubkey::default();
        let key = Pubkey::default();
        let mut lamports = 0;
        let mut data = vec![0; mem::size_of::<CounterAccount>()];
        let owner = program_id;

        let account = AccountInfo::new(
            &key,
            false,
            true,
            &mut lamports,
            &mut data,
            &owner,
            false,
            Epoch::default(),
        );

        let instruction_data: Vec<u8> = vec![0];

        process_instruction(&program_id, &[account.clone()], &instruction_data).unwrap();

        let counter = CounterAccount::try_from_slice(&account.data.borrow()).unwrap();
        assert_eq!(counter.count, 1);
    }
}
```

---

## Anchor Framework

**Anchor** is a framework that simplifies Solana program development by providing:
- High-level abstractions
- Automatic serialization/deserialization
- Account validation
- Error handling
- Testing utilities

### Installation

```bash
# Install Anchor CLI
cargo install --git https://github.com/coral-xyz/anchor avm --locked --force
avm install latest
avm use latest

# Verify installation
anchor --version
```

### Create Anchor Project

```bash
# Create new project
anchor init my-anchor-project
cd my-anchor-project

# Project structure:
# ├── Anchor.toml          # Anchor config
# ├── Cargo.toml           # Rust workspace
# ├── programs/            # Your programs
# │   └── my-anchor-project/
# │       ├── Cargo.toml
# │       └── src/
# │           └── lib.rs
# ├── tests/               # TypeScript tests
# └── migrations/          # Deploy scripts
```

### Anchor Program Structure

```rust
use anchor_lang::prelude::*;

// Declare program ID (replace with your actual program ID after deploy)
declare_id!("Fg6PaFpoGXkYsidMpWTK6W2BeZ7FEfcYkg476zPFsLnS");

#[program]
pub mod my_anchor_project {
    use super::*;

    // Initialize instruction
    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let counter = &mut ctx.accounts.counter;
        counter.count = 0;
        counter.authority = ctx.accounts.user.key();
        msg!("Counter initialized!");
        Ok(())
    }

    // Increment instruction
    pub fn increment(ctx: Context<Update>) -> Result<()> {
        let counter = &mut ctx.accounts.counter;
        counter.count += 1;
        msg!("Counter: {}", counter.count);
        Ok(())
    }

    // Decrement instruction
    pub fn decrement(ctx: Context<Update>) -> Result<()> {
        let counter = &mut ctx.accounts.counter;
        counter.count = counter.count.saturating_sub(1);
        msg!("Counter: {}", counter.count);
        Ok(())
    }
}

// Account structure
#[account]
pub struct Counter {
    pub count: u64,
    pub authority: Pubkey,
}

// Validation structs
#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + 8 + 32,  // discriminator + count + authority
        seeds = [b"counter", user.key().as_ref()],
        bump
    )]
    pub counter: Account<'info, Counter>,

    #[account(mut)]
    pub user: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Update<'info> {
    #[account(
        mut,
        has_one = authority,  // Verify authority matches
        seeds = [b"counter", authority.key().as_ref()],
        bump
    )]
    pub counter: Account<'info, Counter>,

    pub authority: Signer<'info>,
}
```

### Anchor Constraints

Common account constraints:

```rust
#[account(
    init,                          // Create account
    payer = user,                  // Who pays for creation
    space = 8 + 100,              // Account size
    seeds = [b"seed"],            // PDA seeds
    bump,                         // PDA bump seed
    mut,                          // Account is mutable
    has_one = authority,          // Check field matches
    constraint = amount > 0,      // Custom validation
)]
```

### Building and Testing

```bash
# Build
anchor build

# Update program ID in lib.rs and Anchor.toml
anchor keys list

# Test
anchor test

# Deploy
anchor deploy
```

### Anchor Tests (TypeScript)

```typescript
import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { MyAnchorProject } from "../target/types/my_anchor_project";
import { expect } from "chai";

describe("my-anchor-project", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.MyAnchorProject as Program<MyAnchorProject>;
  const user = provider.wallet;

  // Derive PDA
  const [counterPda] = anchor.web3.PublicKey.findProgramAddressSync(
    [Buffer.from("counter"), user.publicKey.toBuffer()],
    program.programId
  );

  it("Initializes counter", async () => {
    const tx = await program.methods
      .initialize()
      .accounts({
        counter: counterPda,
        user: user.publicKey,
        systemProgram: anchor.web3.SystemProgram.programId,
      })
      .rpc();

    const counter = await program.account.counter.fetch(counterPda);
    expect(counter.count.toNumber()).to.equal(0);
    expect(counter.authority.toString()).to.equal(user.publicKey.toString());
  });

  it("Increments counter", async () => {
    await program.methods
      .increment()
      .accounts({
        counter: counterPda,
        authority: user.publicKey,
      })
      .rpc();

    const counter = await program.account.counter.fetch(counterPda);
    expect(counter.count.toNumber()).to.equal(1);
  });

  it("Decrements counter", async () => {
    await program.methods
      .decrement()
      .accounts({
        counter: counterPda,
        authority: user.publicKey,
      })
      .rpc();

    const counter = await program.account.counter.fetch(counterPda);
    expect(counter.count.toNumber()).to.equal(0);
  });
});
```

### Error Handling

```rust
#[error_code]
pub enum ErrorCode {
    #[msg("Amount must be greater than zero")]
    InvalidAmount,

    #[msg("Insufficient funds")]
    InsufficientFunds,

    #[msg("Unauthorized access")]
    Unauthorized,
}

// Use in program
require!(amount > 0, ErrorCode::InvalidAmount);
```

---

## SPL Token Program

**SPL (Solana Program Library)** Token is the standard for fungible and non-fungible tokens on Solana.

### Token Architecture

```
Mint Account (Token Definition)
  ├─→ Token Account 1 (User A's balance)
  ├─→ Token Account 2 (User B's balance)
  └─→ Token Account 3 (User C's balance)
```

### Creating a Token

```bash
# Create a new token mint
spl-token create-token

# Output: Token address (Mint)
# Example: 7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU

# Create token account for yourself
spl-token create-account <TOKEN_ADDRESS>

# Mint tokens
spl-token mint <TOKEN_ADDRESS> 1000

# Check balance
spl-token balance <TOKEN_ADDRESS>

# Transfer tokens
spl-token transfer <TOKEN_ADDRESS> 100 <RECIPIENT_ADDRESS>
```

### Token Account Structure

```rust
pub struct Account {
    pub mint: Pubkey,           // Token type
    pub owner: Pubkey,          // Who owns this account
    pub amount: u64,            // Token balance
    pub delegate: Option<Pubkey>,  // Delegated authority
    pub state: AccountState,    // Initialized/Frozen
    pub is_native: Option<u64>, // Is this wrapped SOL?
    pub delegated_amount: u64,  // Amount delegated
    pub close_authority: Option<Pubkey>,  // Who can close
}
```

### Mint Account Structure

```rust
pub struct Mint {
    pub mint_authority: Option<Pubkey>,  // Who can mint
    pub supply: u64,                     // Total supply
    pub decimals: u8,                    // Decimal places
    pub is_initialized: bool,            // Is initialized?
    pub freeze_authority: Option<Pubkey>, // Who can freeze
}
```

### Associated Token Account (ATA)

**ATA**: Deterministic token account address for each user/mint pair

```rust
// ATA address derived from:
// - User's wallet address
// - Token mint address
// - SPL Token program ID

// Find ATA
let ata = get_associated_token_address(
    &user_wallet,
    &token_mint,
);
```

**Benefits:**
- One account per user per token
- Easy to find user's token account
- No need to track addresses

### Using SPL Token in Anchor

```rust
use anchor_spl::token::{self, Token, TokenAccount, Mint, Transfer};

#[derive(Accounts)]
pub struct TransferTokens<'info> {
    #[account(mut)]
    pub from: Account<'info, TokenAccount>,

    #[account(mut)]
    pub to: Account<'info, TokenAccount>,

    pub authority: Signer<'info>,

    pub token_program: Program<'info, Token>,
}

pub fn transfer_tokens(ctx: Context<TransferTokens>, amount: u64) -> Result<()> {
    let cpi_accounts = Transfer {
        from: ctx.accounts.from.to_account_info(),
        to: ctx.accounts.to.to_account_info(),
        authority: ctx.accounts.authority.to_account_info(),
    };

    let cpi_program = ctx.accounts.token_program.to_account_info();
    let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);

    token::transfer(cpi_ctx, amount)?;
    Ok(())
}
```

### NFT (Non-Fungible Token)

NFT is a token with:
- Supply of 1
- Decimals of 0
- Metadata (Metaplex standard)

```bash
# Create NFT mint (supply=1, decimals=0)
spl-token create-token --decimals 0

# Mint exactly 1 token
spl-token mint <TOKEN_ADDRESS> 1

# Remove mint authority (cannot mint more)
spl-token authorize <TOKEN_ADDRESS> mint --disable
```

---

## Client Development

### Solana Web3.js

```bash
# Install
npm install @solana/web3.js
```

### Basic Connection

```typescript
import {
  Connection,
  PublicKey,
  LAMPORTS_PER_SOL,
  clusterApiUrl,
} from "@solana/web3.js";

// Connect to devnet
const connection = new Connection(clusterApiUrl("devnet"), "confirmed");

// Get balance
const publicKey = new PublicKey("YourPublicKeyHere");
const balance = await connection.getBalance(publicKey);
console.log(`Balance: ${balance / LAMPORTS_PER_SOL} SOL`);

// Get recent blockhash
const { blockhash } = await connection.getLatestBlockhash();

// Get account info
const accountInfo = await connection.getAccountInfo(publicKey);
```

### Sending SOL

```typescript
import {
  Connection,
  Keypair,
  SystemProgram,
  Transaction,
  sendAndConfirmTransaction,
  LAMPORTS_PER_SOL,
} from "@solana/web3.js";

const connection = new Connection(clusterApiUrl("devnet"), "confirmed");

// Load keypair from file or create new
const fromKeypair = Keypair.generate();
const toPublicKey = new PublicKey("RecipientPublicKeyHere");

// Airdrop some SOL for testing
await connection.requestAirdrop(fromKeypair.publicKey, 2 * LAMPORTS_PER_SOL);

// Create transfer instruction
const transaction = new Transaction().add(
  SystemProgram.transfer({
    fromPubkey: fromKeypair.publicKey,
    toPubkey: toPublicKey,
    lamports: 0.1 * LAMPORTS_PER_SOL,
  })
);

// Send transaction
const signature = await sendAndConfirmTransaction(
  connection,
  transaction,
  [fromKeypair]
);

console.log("Transaction signature:", signature);
```

### Interacting with Programs

```typescript
import { Connection, PublicKey, Transaction, TransactionInstruction } from "@solana/web3.js";

const connection = new Connection(clusterApiUrl("devnet"), "confirmed");
const programId = new PublicKey("YourProgramId");
const accountPubkey = new PublicKey("AccountToModify");

// Create instruction
const instruction = new TransactionInstruction({
  keys: [
    { pubkey: accountPubkey, isSigner: false, isWritable: true },
    { pubkey: wallet.publicKey, isSigner: true, isWritable: false },
  ],
  programId,
  data: Buffer.from([0]), // Instruction data
});

// Create and send transaction
const transaction = new Transaction().add(instruction);
const signature = await wallet.sendTransaction(transaction, connection);
await connection.confirmTransaction(signature);
```

### Using Anchor Client

```typescript
import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";

// Load IDL
import idl from "./idl.json";

// Setup provider
const provider = anchor.AnchorProvider.env();
anchor.setProvider(provider);

// Create program interface
const programId = new anchor.web3.PublicKey("YourProgramId");
const program = new Program(idl, programId, provider);

// Call program method
const tx = await program.methods
  .increment()
  .accounts({
    counter: counterPda,
    authority: provider.wallet.publicKey,
  })
  .rpc();

console.log("Transaction signature:", tx);

// Fetch account data
const counter = await program.account.counter.fetch(counterPda);
console.log("Count:", counter.count.toString());
```

### Working with Wallets

```typescript
// Phantom Wallet
const getProvider = () => {
  if ("solana" in window) {
    const provider = window.solana;
    if (provider.isPhantom) {
      return provider;
    }
  }
  window.open("https://phantom.app/", "_blank");
};

// Connect
const provider = getProvider();
const resp = await provider.connect();
console.log("Public key:", resp.publicKey.toString());

// Sign and send transaction
const { signature } = await provider.signAndSendTransaction(transaction);

// Sign message
const message = new TextEncoder().encode("Hello Solana!");
const { signature } = await provider.signMessage(message);
```

---

## CLI Tools

### Solana CLI

#### Keypair Management

```bash
# Generate new keypair
solana-keygen new

# Recover from seed phrase
solana-keygen recover

# Show public key
solana-keygen pubkey ~/.config/solana/id.json

# Verify keypair
solana-keygen verify <PUBKEY> ~/.config/solana/id.json
```

#### Configuration

```bash
# Show config
solana config get

# Set RPC URL
solana config set --url https://api.devnet.solana.com
solana config set --url https://api.mainnet-beta.solana.com
solana config set --url http://localhost:8899  # Local

# Set keypair
solana config set --keypair ~/.config/solana/id.json
```

#### Account Operations

```bash
# Get balance
solana balance

# Get account info
solana account <ADDRESS>

# Airdrop (devnet/testnet only)
solana airdrop 2

# Transfer SOL
solana transfer <RECIPIENT> 0.5

# Check transaction
solana confirm <SIGNATURE>
```

#### Program Operations

```bash
# Deploy program
solana program deploy /path/to/program.so

# Show program
solana program show <PROGRAM_ID>

# Get program account data
solana account <PROGRAM_ID>

# Close program (recovers rent)
solana program close <PROGRAM_ID>

# Upgrade program
solana program deploy --program-id <PROGRAM_ID> /path/to/new_program.so
```

### SPL Token CLI

```bash
# Create new token
spl-token create-token
spl-token create-token --decimals 9

# Create token account
spl-token create-account <TOKEN_MINT>

# Get token accounts
spl-token accounts

# Mint tokens
spl-token mint <TOKEN_MINT> <AMOUNT>

# Transfer tokens
spl-token transfer <TOKEN_MINT> <AMOUNT> <RECIPIENT>

# Get token supply
spl-token supply <TOKEN_MINT>

# Burn tokens
spl-token burn <TOKEN_ACCOUNT> <AMOUNT>

# Authorize operations
spl-token authorize <TOKEN_MINT> mint <NEW_AUTHORITY>
spl-token authorize <TOKEN_MINT> mint --disable  # Remove mint authority

# Close token account (get rent back)
spl-token close <TOKEN_ACCOUNT>

# Wrap SOL (create wSOL)
spl-token wrap 1.0

# Unwrap SOL
spl-token unwrap
```

### Local Validator

```bash
# Start local validator
solana-test-validator

# Start with specific programs
solana-test-validator --bpf-program <PROGRAM_ID> /path/to/program.so

# Reset ledger
solana-test-validator --reset

# Clone account from mainnet
solana-test-validator --clone <ADDRESS>

# Set compute units
solana-test-validator --compute-unit-limit 200000
```

---

## Development Tools and Resources

### IDEs and Editors

#### 1. VS Code Extensions
- **Rust Analyzer**: Rust language support
- **Solana**: Syntax highlighting for Solana programs
- **Better TOML**: TOML file support

```bash
# Install Rust Analyzer
code --install-extension rust-lang.rust-analyzer
```

#### 2. Solana Playground
- Browser-based IDE
- No installation required
- Built-in wallet
- URL: https://beta.solpg.io

### Testing Tools

#### Local Validator

```bash
# Install
cargo install --git https://github.com/solana-labs/solana solana-test-validator

# Run with logs
solana-test-validator -l

# View logs
solana logs
```

#### Anchor Test

```bash
# Run all tests
anchor test

# Run specific test
anchor test --skip-local-validator
```

### Explorers

#### Solana Explorer
- **Mainnet**: https://explorer.solana.com
- **Devnet**: https://explorer.solana.com/?cluster=devnet
- **Testnet**: https://explorer.solana.com/?cluster=testnet
- View transactions, accounts, blocks
- Decode program instructions

#### Solscan
- URL: https://solscan.io
- Advanced analytics
- Token information
- Program interactions

#### Solana Beach
- URL: https://solanabeach.io
- Validator information
- Network statistics

### Faucets (Testnet SOL)

```bash
# Via CLI (devnet)
solana airdrop 2

# Web faucets:
# - https://faucet.solana.com
# - https://solfaucet.com
```

### Libraries and SDKs

#### Rust
```toml
[dependencies]
solana-program = "1.18"
solana-sdk = "1.18"
anchor-lang = "0.29"
anchor-spl = "0.29"
borsh = "0.10"
```

#### JavaScript/TypeScript
```bash
npm install @solana/web3.js
npm install @coral-xyz/anchor
npm install @solana/spl-token
npm install @metaplex-foundation/js
```

#### Python
```bash
pip install solana
pip install anchorpy
```

### Important Programs

#### Native Programs
- **System Program**: `11111111111111111111111111111111`
- **Token Program**: `TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA`
- **Associated Token Account**: `ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL`
- **Rent**: `SysvarRent111111111111111111111111111111111`

### Useful Resources

#### Documentation
- **Solana Docs**: https://docs.solana.com
- **Anchor Docs**: https://www.anchor-lang.com
- **SPL Token**: https://spl.solana.com/token
- **Solana Cookbook**: https://solanacookbook.com

#### Learning Platforms
- **Solana Bootcamp**: Official developer program
- **buildspace**: Project-based tutorials
- **Questbook**: Hands-on learning
- **Ackee School**: Security auditing course

#### Community
- **Discord**: Solana official server
- **Stack Exchange**: Solana Q&A
- **GitHub**: https://github.com/solana-labs
- **Twitter**: @solana, @SolanaDevs

### Development Checklist

- [ ] Install Rust and Solana CLI
- [ ] Install Anchor framework
- [ ] Set up local validator
- [ ] Create and fund devnet wallet
- [ ] Write and test program locally
- [ ] Deploy to devnet
- [ ] Test with frontend integration
- [ ] Security audit
- [ ] Deploy to mainnet
- [ ] Monitor program

### Common Pitfalls

1. **Forgetting rent exemption**
   - Solution: Always calculate and fund rent-exempt balance

2. **Account ownership confusion**
   - Solution: Remember only owner program can modify account data

3. **PDA seed mismatch**
   - Solution: Use same seeds consistently for derivation

4. **Missing account in instruction**
   - Solution: Include all accounts needed (even read-only)

5. **Compute budget exceeded**
   - Solution: Optimize code or request more compute units

6. **Serialization errors**
   - Solution: Use consistent serialization (Borsh recommended)

---

## Performance Optimization

### Compute Units

Each transaction has compute unit limit (default: 200,000)

```rust
// Request more compute units
solana_program::compute_budget::ComputeBudgetInstruction::set_compute_unit_limit(400_000);
```

### Account Size

Minimize account size to reduce rent:

```rust
// Bad: Large struct with unused fields
#[account]
pub struct BadAccount {
    pub data: [u8; 10000],  // Wastes space
}

// Good: Only store what's needed
#[account]
pub struct GoodAccount {
    pub count: u64,
    pub owner: Pubkey,
}
```

### Zero-Copy Deserialization

For large accounts, use zero-copy:

```rust
#[account(zero_copy)]
pub struct LargeAccount {
    pub data: [u8; 10000],
}
```

### Parallel Transactions

Structure transactions to avoid account conflicts:

```typescript
// These can run in parallel (different accounts)
const tx1 = transfer(accountA, accountB);
const tx2 = transfer(accountC, accountD);

// Send simultaneously
await Promise.all([
  connection.sendTransaction(tx1),
  connection.sendTransaction(tx2),
]);
```

---

## Security Best Practices

### 1. Account Validation

```rust
// Always validate account ownership
if account.owner != program_id {
    return Err(ProgramError::IncorrectProgramId);
}

// Verify signer
if !account.is_signer {
    return Err(ProgramError::MissingRequiredSignature);
}

// Verify account is writable
if !account.is_writable {
    return Err(ProgramError::InvalidArgument);
}
```

### 2. Overflow Protection

```rust
// Use checked math
let result = amount.checked_add(increase)
    .ok_or(ProgramError::ArithmeticOverflow)?;

// Or saturating math
let result = amount.saturating_add(increase);
```

### 3. Signer Authorization

```rust
#[derive(Accounts)]
pub struct Transfer<'info> {
    #[account(mut, has_one = authority)]
    pub account: Account<'info, MyAccount>,

    pub authority: Signer<'info>,  // Must sign
}
```

### 4. PDA Verification

```rust
// Verify PDA derivation
let (expected_pda, bump) = Pubkey::find_program_address(
    &[b"seed", user.key().as_ref()],
    program_id
);

require_keys_eq!(account.key(), expected_pda, ErrorCode::InvalidPDA);
```

### 5. Reentrancy Protection

Solana's account locking prevents reentrancy, but still be careful with CPIs:

```rust
// Anchor provides automatic reentrancy protection
// But be careful with state updates before CPIs
ctx.accounts.user_account.balance -= amount;  // Update first
token::transfer(cpi_ctx, amount)?;            // Then CPI
```

---

**Last Updated**: 2025-01-19
