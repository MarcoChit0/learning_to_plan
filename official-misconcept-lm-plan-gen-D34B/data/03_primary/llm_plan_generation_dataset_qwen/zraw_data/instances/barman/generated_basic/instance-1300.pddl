(define (problem prob)
 (:domain barman)
 (:objects 
      shaker482 - shaker
      left right - hand
      shot130 shot450 shot307 - shot
      ingredient452 ingredient24 - ingredient
      cocktail486 - cocktail
      dispenser334 dispenser208 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker482)
  (ontable shot130)
  (ontable shot450)
  (ontable shot307)
  (dispenses dispenser334 ingredient452)
  (dispenses dispenser208 ingredient24)
  (clean shaker482)
  (clean shot130)
  (clean shot450)
  (clean shot307)
  (empty shaker482)
  (empty shot130)
  (empty shot450)
  (empty shot307)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker482 l0)
  (shaker-level shaker482 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail486 ingredient24)
  (cocktail-part2 cocktail486 ingredient452)
)
 (:goal
  (and
      (contains shot130 cocktail486)
      (contains shot450 cocktail486)
)))
