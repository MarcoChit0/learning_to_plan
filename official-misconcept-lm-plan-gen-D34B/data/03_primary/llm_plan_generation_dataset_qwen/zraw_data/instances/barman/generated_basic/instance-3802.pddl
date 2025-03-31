(define (problem prob)
 (:domain barman)
 (:objects 
      shaker9 - shaker
      left right - hand
      shot21 - shot
      ingredient285 ingredient279 ingredient267 ingredient230 - ingredient
      cocktail190 - cocktail
      dispenser283 dispenser484 dispenser73 dispenser2 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker9)
  (ontable shot21)
  (dispenses dispenser283 ingredient285)
  (dispenses dispenser484 ingredient279)
  (dispenses dispenser73 ingredient267)
  (dispenses dispenser2 ingredient230)
  (clean shaker9)
  (clean shot21)
  (empty shaker9)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker9 l0)
  (shaker-level shaker9 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient279)
  (cocktail-part2 cocktail190 ingredient285)
)
 (:goal
  (and
      (contains shot21 cocktail190)
)))
