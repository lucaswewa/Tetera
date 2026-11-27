//! Devices' events: each subscriber gets every event emitted after it subscribed, on a channel
//! of its own; a subscriber that has gone is let go on the next event.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Mutex, PoisonError};

/// The subscribers to a device's events.
#[derive(Debug)]
pub struct Events<E> {
    subscribers: Mutex<Vec<Sender<E>>>,
}

impl<E> Default for Events<E> {
    fn default() -> Self {
        Self {
            subscribers: Mutex::new(Vec::new()),
        }
    }
}

impl<E: Clone> Events<E> {
    /// A new subscriber's channel.
    pub fn subscribe(&self) -> Receiver<E> {
        let (tx, rx) = channel();
        self.subscribers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(tx);
        rx
    }

    /// `event` to every subscriber still listening.
    pub fn emit(&self, event: E) {
        self.subscribers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|tx| tx.send(event.clone()).is_ok());
    }

    /// How many are listening, as of the last event.
    pub fn subscribers(&self) -> usize {
        self.subscribers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_subscriber_gets_the_events_after_it_came_and_the_gone_are_let_go() {
        let events = Events::default();
        events.emit(0);
        let a = events.subscribe();
        events.emit(1);
        let b = events.subscribe();
        events.emit(2);
        assert_eq!(a.try_iter().collect::<Vec<_>>(), [1, 2]);
        assert_eq!(b.try_iter().collect::<Vec<_>>(), [2]);
        assert_eq!(events.subscribers(), 2);
        drop(a);
        events.emit(3);
        assert_eq!(events.subscribers(), 1);
        assert_eq!(b.try_iter().collect::<Vec<_>>(), [3]);
    }
}
