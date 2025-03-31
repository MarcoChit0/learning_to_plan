(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot486 shot73 shot212 - shot
      ingredient215 ingredient195 ingredient261 - ingredient
      cocktail482 - cocktail
      dispenser285 dispenser280 dispenser444 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot486)
  (ontable shot73)
  (ontable shot212)
  (dispenses dispenser285 ingredient215)
  (dispenses dispenser280 ingredient195)
  (dispenses dispenser444 ingredient261)
  (clean shaker257)
  (clean shot486)
  (clean shot73)
  (clean shot212)
  (empty shaker257)
  (empty shot486)
  (empty shot73)
  (empty shot212)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient261)
  (cocktail-part2 cocktail482 ingredient215)
)
 (:goal
  (and
      (contains shot486 cocktail482)
      (contains shot73 cocktail482)
)))
