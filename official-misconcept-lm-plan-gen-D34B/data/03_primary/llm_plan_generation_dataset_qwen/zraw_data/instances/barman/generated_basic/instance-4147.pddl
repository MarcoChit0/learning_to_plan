(define (problem prob)
 (:domain barman)
 (:objects 
      shaker65 - shaker
      left right - hand
      shot314 - shot
      ingredient171 ingredient9 - ingredient
      cocktail482 - cocktail
      dispenser22 dispenser286 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker65)
  (ontable shot314)
  (dispenses dispenser22 ingredient171)
  (dispenses dispenser286 ingredient9)
  (clean shaker65)
  (clean shot314)
  (empty shaker65)
  (empty shot314)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker65 l0)
  (shaker-level shaker65 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient9)
  (cocktail-part2 cocktail482 ingredient171)
)
 (:goal
  (and
      (contains shot314 cocktail482)
)))
