(define (problem prob)
 (:domain barman)
 (:objects 
      shaker385 - shaker
      left right - hand
      shot397 shot373 - shot
      ingredient177 ingredient167 ingredient364 - ingredient
      cocktail1 - cocktail
      dispenser220 dispenser361 dispenser341 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker385)
  (ontable shot397)
  (ontable shot373)
  (dispenses dispenser220 ingredient177)
  (dispenses dispenser361 ingredient167)
  (dispenses dispenser341 ingredient364)
  (clean shaker385)
  (clean shot397)
  (clean shot373)
  (empty shaker385)
  (empty shot397)
  (empty shot373)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker385 l0)
  (shaker-level shaker385 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient167)
  (cocktail-part2 cocktail1 ingredient364)
)
 (:goal
  (and
      (contains shot397 cocktail1)
)))
