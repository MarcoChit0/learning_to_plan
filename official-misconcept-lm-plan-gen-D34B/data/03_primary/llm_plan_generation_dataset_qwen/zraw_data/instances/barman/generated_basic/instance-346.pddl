(define (problem prob)
 (:domain barman)
 (:objects 
      shaker172 - shaker
      left right - hand
      shot485 shot64 - shot
      ingredient57 ingredient439 ingredient159 ingredient367 - ingredient
      cocktail1 - cocktail
      dispenser204 dispenser320 dispenser265 dispenser382 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker172)
  (ontable shot485)
  (ontable shot64)
  (dispenses dispenser204 ingredient57)
  (dispenses dispenser320 ingredient439)
  (dispenses dispenser265 ingredient159)
  (dispenses dispenser382 ingredient367)
  (clean shaker172)
  (clean shot485)
  (clean shot64)
  (empty shaker172)
  (empty shot485)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker172 l0)
  (shaker-level shaker172 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient367)
  (cocktail-part2 cocktail1 ingredient57)
)
 (:goal
  (and
      (contains shot485 cocktail1)
)))
