(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot317 - shot
      ingredient329 ingredient290 ingredient115 ingredient499 - ingredient
      cocktail453 - cocktail
      dispenser298 dispenser289 dispenser419 dispenser294 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot317)
  (dispenses dispenser298 ingredient329)
  (dispenses dispenser289 ingredient290)
  (dispenses dispenser419 ingredient115)
  (dispenses dispenser294 ingredient499)
  (clean shaker24)
  (clean shot317)
  (empty shaker24)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail453 ingredient115)
  (cocktail-part2 cocktail453 ingredient290)
)
 (:goal
  (and
      (contains shot317 cocktail453)
)))
