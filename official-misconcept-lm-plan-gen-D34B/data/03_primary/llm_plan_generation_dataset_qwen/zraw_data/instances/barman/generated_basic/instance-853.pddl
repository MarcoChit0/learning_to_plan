(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot398 - shot
      ingredient41 ingredient184 ingredient46 - ingredient
      cocktail1 - cocktail
      dispenser280 dispenser263 dispenser184 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot398)
  (dispenses dispenser280 ingredient41)
  (dispenses dispenser263 ingredient184)
  (dispenses dispenser184 ingredient46)
  (clean shaker37)
  (clean shot398)
  (empty shaker37)
  (empty shot398)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient46)
  (cocktail-part2 cocktail1 ingredient184)
)
 (:goal
  (and
      (contains shot398 cocktail1)
)))
