//! Tiles of the H&E sample (P1.11b, ADR-0015): keyed by level of detail, column, row and
//! depth, and kept in an LRU cache within a budget of bytes. The cache doesn't care what a
//! tile holds, so it is tested here without a GPU; on the GPU, a tile is its stains in half
//! precision.

use crate::render::{Depth, LAYER, Region};
use std::collections::HashMap;
use tetera_core::{Error, Result};

/// A tile's side, pixels.
pub const TILE: u32 = 256;
/// Level 0's pixel, µm: finer than a 50× objective's sampling of 0.048 µm.
pub const FINEST: f64 = 0.04;
/// The coarsest level.
pub const COARSEST: u8 = 15;

/// Which depths a tile sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TileDepth {
    /// The whole section.
    Whole,
    /// `count` layers of 0.5 µm from the `first`, above the section's lower face.
    Layers {
        /// The first.
        first: u8,
        /// How many.
        count: u8,
    },
}

/// A tile's key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileKey {
    /// Its level of detail: level ℓ's pixel is 0.04 × 2^ℓ µm.
    pub level: u8,
    /// Its column on the slide, tiles from the origin.
    pub column: i32,
    /// Its row.
    pub row: i32,
    /// The depths it sums.
    pub depth: TileDepth,
}

impl TileKey {
    /// Its pixel, µm.
    pub fn pixel(&self) -> f64 {
        FINEST * f64::from(1u32 << self.level.min(COARSEST))
    }

    /// The region it covers.
    pub fn region(&self) -> Region {
        let side = f64::from(TILE) * self.pixel();
        let depth = match self.depth {
            TileDepth::Whole => Depth::Whole,
            TileDepth::Layers { first, count } => Depth::Range(
                f32::from(first) * LAYER,
                (f32::from(first) + f32::from(count)) * LAYER,
            ),
        };
        Region {
            origin: (f64::from(self.column) * side, f64::from(self.row) * side),
            width: TILE,
            height: TILE,
            pixel: self.pixel(),
            depth,
        }
    }

    /// The level just finer than `pixel` µm, or the coarsest, from which a camera's sampling
    /// is resampled.
    pub fn level_for(pixel: f64) -> u8 {
        (0..=COARSEST)
            .rev()
            .find(|&level| FINEST * f64::from(1u32 << level) <= pixel)
            .unwrap_or(0)
    }
}

/// What a cache has done.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Tiles found.
    pub hits: u64,
    /// Tiles made.
    pub misses: u64,
    /// Tiles let go to stay within the budget.
    pub evictions: u64,
    /// Tiles held.
    pub tiles: usize,
    /// Bytes they hold.
    pub bytes: u64,
}

/// An LRU cache of tiles within a budget of bytes, each tile the same size.
#[derive(Debug)]
pub struct TileCache<V> {
    budget: u64,
    tile_bytes: u64,
    tiles: HashMap<TileKey, (V, u64)>,
    clock: u64,
    stats: CacheStats,
}

impl<V: Clone> TileCache<V> {
    /// A cache of tiles of `tile_bytes` each, within `budget` bytes: room for one at least.
    pub fn new(budget: u64, tile_bytes: u64) -> Result<Self> {
        if tile_bytes == 0 || budget < tile_bytes {
            return Err(Error::invalid_argument(format!(
                "a budget of {budget} bytes for tiles of {tile_bytes}: it holds one at least"
            )));
        }
        Ok(Self {
            budget,
            tile_bytes,
            tiles: HashMap::new(),
            clock: 0,
            stats: CacheStats::default(),
        })
    }

    /// How many tiles it holds at most.
    pub fn capacity(&self) -> u64 {
        self.budget / self.tile_bytes
    }

    /// The tile for `key`: found, or made by `make`, letting the least recently used go
    /// first if the budget is full. A tile `make` fails to make isn't kept.
    pub fn get_or_make(
        &mut self,
        key: TileKey,
        make: impl FnOnce(&TileKey) -> Result<V>,
    ) -> Result<V> {
        self.clock += 1;
        if let Some((tile, used)) = self.tiles.get_mut(&key) {
            *used = self.clock;
            self.stats.hits += 1;
            return Ok(tile.clone());
        }
        while self.tiles.len() as u64 >= self.capacity() {
            let oldest = *self
                .tiles
                .iter()
                // The least recently used; of a tie, which can't happen, the least key.
                .min_by_key(|(key, (_, used))| (*used, **key))
                .map(|(key, _)| key)
                .expect("a full cache holds a tile");
            self.tiles.remove(&oldest);
            self.stats.evictions += 1;
        }
        let tile = make(&key)?;
        self.tiles.insert(key, (tile.clone(), self.clock));
        self.stats.misses += 1;
        Ok(tile)
    }

    /// What it has done, and holds.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            tiles: self.tiles.len(),
            bytes: self.tiles.len() as u64 * self.tile_bytes,
            ..self.stats
        }
    }

    /// Whether it holds `key`'s tile.
    pub fn holds(&self, key: &TileKey) -> bool {
        self.tiles.contains_key(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(column: i32) -> TileKey {
        TileKey {
            level: 3,
            column,
            row: -2,
            depth: TileDepth::Whole,
        }
    }

    #[test]
    fn a_tile_covers_its_place_at_its_level() {
        let k = TileKey {
            level: 3,
            column: 2,
            row: -1,
            depth: TileDepth::Layers { first: 5, count: 3 },
        };
        assert!((k.pixel() - 0.32).abs() < 1e-12);
        let r = k.region();
        assert_eq!((r.width, r.height), (256, 256));
        assert!((r.origin.0 - 163.84).abs() < 1e-9 && (r.origin.1 + 81.92).abs() < 1e-9);
        assert_eq!(r.depth, Depth::Range(2.5, 4.0));
        assert_eq!(key(0).region().depth, Depth::Whole);
        let coarsest = TileKey { level: 200, ..k };
        assert_eq!(
            coarsest.pixel(),
            FINEST * 32_768.0,
            "no coarser than the coarsest"
        );
    }

    #[test]
    fn a_sampling_takes_the_level_just_finer() {
        assert_eq!(TileKey::level_for(0.048), 0, "50×");
        assert_eq!(TileKey::level_for(0.12), 1, "20×: 0.08 µm");
        assert_eq!(TileKey::level_for(0.48), 3, "5×: 0.32 µm");
        assert_eq!(TileKey::level_for(0.08), 1, "exactly a level");
        assert_eq!(TileKey::level_for(0.01), 0, "finer than the finest");
        assert_eq!(TileKey::level_for(1e9), COARSEST);
    }

    #[test]
    fn a_budget_holds_one_tile_at_least() {
        assert!(TileCache::<u8>::new(99, 100).is_err());
        assert!(TileCache::<u8>::new(100, 0).is_err());
        assert_eq!(TileCache::<u8>::new(350, 100).unwrap().capacity(), 3);
    }

    #[test]
    fn the_least_recently_used_tile_goes_first() {
        let mut cache = TileCache::new(300, 100).unwrap();
        let made = std::cell::Cell::new(0);
        let get = |cache: &mut TileCache<i32>, column| {
            cache
                .get_or_make(key(column), |k| {
                    made.set(made.get() + 1);
                    Ok(k.column * 10)
                })
                .unwrap()
        };
        assert_eq!(
            (get(&mut cache, 1), get(&mut cache, 2), get(&mut cache, 3)),
            (10, 20, 30)
        );
        assert_eq!(get(&mut cache, 1), 10, "found, and now the most recent");
        get(&mut cache, 4);
        assert!(!cache.holds(&key(2)), "the least recent went");
        assert!(cache.holds(&key(1)) && cache.holds(&key(3)) && cache.holds(&key(4)));
        get(&mut cache, 5);
        assert!(!cache.holds(&key(3)));
        assert_eq!(
            cache.stats(),
            CacheStats {
                hits: 1,
                misses: 5,
                evictions: 2,
                tiles: 3,
                bytes: 300
            }
        );
        assert_eq!(made.get(), 5);
    }

    #[test]
    fn a_tile_that_fails_to_be_made_is_not_kept() {
        let mut cache = TileCache::<i32>::new(200, 100).unwrap();
        assert!(
            cache
                .get_or_make(key(1), |_| Err(Error::hardware("no")))
                .is_err()
        );
        assert!(!cache.holds(&key(1)));
        assert_eq!(cache.get_or_make(key(1), |_| Ok(7)).unwrap(), 7);
        assert_eq!(cache.stats().misses, 1);
    }
}
