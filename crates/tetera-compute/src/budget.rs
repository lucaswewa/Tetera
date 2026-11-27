//! VRAM budgets (ADR-0010): the GPU's memory shared out between its users, the simulator's
//! tile cache, the preview and compute jobs among them. Each reserves a budget; an
//! allocation beyond its own budget is refused, so no user can starve another. A grant
//! holds part of a budget until it drops.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use tetera_core::{Error, Result};

const MIB: f64 = 1_048_576.0;

fn mib(bytes: u64) -> String {
    format!("{:.0} MiB", bytes as f64 / MIB)
}

#[derive(Debug)]
struct Account {
    limit: u64,
    used: u64,
    open: bool,
}

#[derive(Debug)]
struct State {
    capacity: u64,
    accounts: BTreeMap<String, Account>,
}

impl State {
    fn reserved(&self) -> u64 {
        self.accounts.values().map(|a| a.limit).sum()
    }

    /// Removes `user`'s account once its budget is released and nothing is granted from it.
    fn tidy(&mut self, user: &str) {
        if self
            .accounts
            .get(user)
            .is_some_and(|a| !a.open && a.used == 0)
        {
            self.accounts.remove(user);
        }
    }
}

type Shared = Arc<Mutex<State>>;

fn lock(state: &Shared) -> MutexGuard<'_, State> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The GPU's memory, shared out as budgets. Cloning it shares the same accounts.
#[derive(Debug, Clone)]
pub struct VramBudgets {
    state: Shared,
}

/// One user's budget: reserved until it drops, and its grants with it.
#[derive(Debug)]
pub struct Budget {
    user: String,
    state: Shared,
}

/// Part of a budget, held until it drops.
#[derive(Debug)]
pub struct Grant {
    user: String,
    bytes: u64,
    state: Shared,
}

/// One account, as [`VramBudgets::accounts`] reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccountReport {
    /// Whose it is.
    pub user: String,
    /// Its limit, in bytes.
    pub limit: u64,
    /// Granted from it now, in bytes.
    pub used: u64,
}

impl VramBudgets {
    /// Budgets that may reserve `capacity` bytes in all.
    pub fn new(capacity: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                capacity,
                accounts: BTreeMap::new(),
            })),
        }
    }

    /// Budgets for a device with `free` bytes, keeping `keep_free` of them (a fraction from
    /// 0 to 1) out of every budget, for the driver and other programs.
    pub fn from_free(free: u64, keep_free: f64) -> Self {
        let share = (1.0 - keep_free.clamp(0.0, 1.0)) * free as f64;
        Self::new(share as u64)
    }

    /// The bytes that may be reserved in all.
    pub fn capacity(&self) -> u64 {
        lock(&self.state).capacity
    }

    /// The bytes reserved now.
    pub fn reserved(&self) -> u64 {
        lock(&self.state).reserved()
    }

    /// Reserves `limit` bytes for `user`, unless that would pass the capacity or `user`
    /// already has a budget.
    pub fn reserve(&self, user: &str, limit: u64) -> Result<Budget> {
        let mut state = lock(&self.state);
        if state.accounts.contains_key(user) {
            return Err(Error::invalid_argument(format!(
                "{user} already has a VRAM budget"
            )));
        }
        let free = state.capacity.saturating_sub(state.reserved());
        if limit > free {
            return Err(Error::busy(format!(
                "{user} asks for {} of VRAM, but only {} of {} are unreserved",
                mib(limit),
                mib(free),
                mib(state.capacity)
            )));
        }
        state.accounts.insert(
            user.to_string(),
            Account {
                limit,
                used: 0,
                open: true,
            },
        );
        Ok(Budget {
            user: user.to_string(),
            state: self.state.clone(),
        })
    }

    /// Every account, by user.
    pub fn accounts(&self) -> Vec<AccountReport> {
        lock(&self.state)
            .accounts
            .iter()
            .map(|(user, a)| AccountReport {
                user: user.clone(),
                limit: a.limit,
                used: a.used,
            })
            .collect()
    }
}

impl Budget {
    /// Whose budget it is.
    pub fn user(&self) -> &str {
        &self.user
    }

    /// Its limit, in bytes.
    pub fn limit(&self) -> u64 {
        lock(&self.state)
            .accounts
            .get(&self.user)
            .map_or(0, |a| a.limit)
    }

    /// Granted from it now, in bytes.
    pub fn used(&self) -> u64 {
        lock(&self.state)
            .accounts
            .get(&self.user)
            .map_or(0, |a| a.used)
    }

    /// Grants `bytes` from the budget, unless they won't fit in what is left of it.
    pub fn allocate(&self, bytes: u64) -> Result<Grant> {
        let mut state = lock(&self.state);
        let account = state
            .accounts
            .get_mut(&self.user)
            .ok_or_else(|| Error::internal(format!("{}'s budget has no account", self.user)))?;
        let left = account.limit - account.used;
        if bytes > left {
            return Err(Error::busy(format!(
                "{}'s VRAM budget of {} has {} left: {} won't fit",
                self.user,
                mib(account.limit),
                mib(left),
                mib(bytes)
            )));
        }
        account.used += bytes;
        Ok(Grant {
            user: self.user.clone(),
            bytes,
            state: self.state.clone(),
        })
    }
}

impl Drop for Budget {
    fn drop(&mut self) {
        let mut state = lock(&self.state);
        if let Some(account) = state.accounts.get_mut(&self.user) {
            account.open = false;
        }
        state.tidy(&self.user);
    }
}

impl Grant {
    /// Its size, in bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Grant {
    fn drop(&mut self) {
        let mut state = lock(&self.state);
        if let Some(account) = state.accounts.get_mut(&self.user) {
            account.used -= self.bytes;
        }
        state.tidy(&self.user);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn budgets_share_out_the_capacity_and_no_more() {
        let budgets = VramBudgets::new(4 * GIB);
        let sim = budgets.reserve("simulator", 3 * GIB).unwrap();
        let err = budgets.reserve("preview", 2 * GIB).unwrap_err();
        assert_eq!(err.kind(), tetera_core::ErrorKind::Busy);
        assert_eq!(
            err.to_string(),
            "preview asks for 2048 MiB of VRAM, but only 1024 MiB of 4096 MiB are unreserved"
        );
        let preview = budgets.reserve("preview", GIB).unwrap();
        assert_eq!((budgets.reserved(), budgets.capacity()), (4 * GIB, 4 * GIB));
        assert_eq!(
            (sim.user(), sim.limit(), preview.limit()),
            ("simulator", 3 * GIB, GIB)
        );
        assert!(budgets.reserve("jobs", 1).is_err(), "nothing is left");
        assert!(
            budgets.reserve("jobs", 0).is_ok(),
            "an empty budget always fits"
        );
    }

    #[test]
    fn a_user_has_one_budget_at_a_time() {
        let budgets = VramBudgets::new(GIB);
        let first = budgets.reserve("preview", 1).unwrap();
        let err = budgets.reserve("preview", 1).unwrap_err();
        assert_eq!(err.to_string(), "preview already has a VRAM budget");
        drop(first);
        assert!(
            budgets.reserve("preview", 1).is_ok(),
            "once released, it can be reserved again"
        );
    }

    #[test]
    fn an_allocation_beyond_its_budget_is_refused() {
        let budgets = VramBudgets::new(4 * GIB);
        let jobs = budgets.reserve("jobs", GIB).unwrap();
        let a = jobs.allocate(600 << 20).unwrap();
        let err = jobs.allocate(500 << 20).unwrap_err();
        assert_eq!(err.kind(), tetera_core::ErrorKind::Busy);
        assert_eq!(
            err.to_string(),
            "jobs's VRAM budget of 1024 MiB has 424 MiB left: 500 MiB won't fit"
        );
        let b = jobs.allocate(424 << 20).unwrap();
        assert_eq!(
            (jobs.used(), a.bytes(), b.bytes()),
            (GIB, 600 << 20, 424 << 20)
        );
        assert!(jobs.allocate(1).is_err(), "full to the byte");
        drop(a);
        assert_eq!(
            jobs.used(),
            424 << 20,
            "a grant gives its bytes back when it drops"
        );
        assert!(jobs.allocate(600 << 20).is_ok());
    }

    #[test]
    fn a_released_budget_stays_reserved_while_its_grants_are_held() {
        let budgets = VramBudgets::new(2 * GIB);
        let sim = budgets.reserve("simulator", 2 * GIB).unwrap();
        let tile = sim.allocate(GIB).unwrap();
        drop(sim);
        assert_eq!(budgets.reserved(), 2 * GIB, "the tile still holds memory");
        assert!(budgets.reserve("preview", 1).is_err());
        assert_eq!(
            budgets.accounts(),
            [AccountReport {
                user: "simulator".into(),
                limit: 2 * GIB,
                used: GIB
            }]
        );
        drop(tile);
        assert_eq!(budgets.reserved(), 0);
        assert!(budgets.accounts().is_empty());
        assert!(budgets.reserve("preview", 2 * GIB).is_ok());
    }

    #[test]
    fn the_capacity_keeps_a_share_free() {
        assert_eq!(
            VramBudgets::from_free(10 * GIB, 0.25).capacity(),
            7 * GIB + GIB / 2
        );
        assert_eq!(
            VramBudgets::from_free(GIB, 1.5).capacity(),
            0,
            "a share beyond 1 counts as 1"
        );
        assert_eq!(
            VramBudgets::from_free(GIB, -1.0).capacity(),
            GIB,
            "below 0 counts as 0"
        );
    }

    #[test]
    fn grants_from_many_threads_never_pass_the_limit() {
        let budgets = VramBudgets::new(GIB);
        let jobs = budgets.reserve("jobs", 100).unwrap();
        let granted = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..1_000 {
                        if let Ok(_grant) = jobs.allocate(30) {
                            granted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let used = jobs.used();
                            assert!(
                                (30..=90).contains(&used),
                                "{used} used: at most three grants of 30 fit"
                            );
                        }
                    }
                });
            }
        });
        assert_eq!(jobs.used(), 0, "every grant gave its bytes back");
        assert!(
            granted.into_inner() >= 1,
            "the first allocation finds the budget empty"
        );
    }
}
