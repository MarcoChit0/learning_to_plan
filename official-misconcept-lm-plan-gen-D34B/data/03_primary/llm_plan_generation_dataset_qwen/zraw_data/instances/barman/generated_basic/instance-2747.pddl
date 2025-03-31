(define (problem prob)
 (:domain barman)
 (:objects 
      shaker471 - shaker
      left right - hand
      shot178 - shot
      ingredient156 ingredient452 - ingredient
      cocktail419 - cocktail
      dispenser317 dispenser50 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker471)
  (ontable shot178)
  (dispenses dispenser317 ingredient156)
  (dispenses dispenser50 ingredient452)
  (clean shaker471)
  (clean shot178)
  (empty shaker471)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker471 l0)
  (shaker-level shaker471 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail419 ingredient452)
  (cocktail-part2 cocktail419 ingredient156)
)
 (:goal
  (and
      (contains shot178 cocktail419)
)))
