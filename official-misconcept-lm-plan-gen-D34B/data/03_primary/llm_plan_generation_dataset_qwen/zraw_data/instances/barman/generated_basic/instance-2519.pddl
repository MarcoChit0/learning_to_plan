(define (problem prob)
 (:domain barman)
 (:objects 
      shaker315 - shaker
      left right - hand
      shot285 - shot
      ingredient292 ingredient245 - ingredient
      cocktail290 - cocktail
      dispenser484 dispenser378 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker315)
  (ontable shot285)
  (dispenses dispenser484 ingredient292)
  (dispenses dispenser378 ingredient245)
  (clean shaker315)
  (clean shot285)
  (empty shaker315)
  (empty shot285)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker315 l0)
  (shaker-level shaker315 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail290 ingredient245)
  (cocktail-part2 cocktail290 ingredient292)
)
 (:goal
  (and
      (contains shot285 cocktail290)
)))
