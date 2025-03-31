(define (problem prob)
 (:domain barman)
 (:objects 
      shaker16 - shaker
      left right - hand
      shot390 shot373 - shot
      ingredient256 ingredient180 ingredient19 - ingredient
      cocktail337 - cocktail
      dispenser403 dispenser147 dispenser362 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker16)
  (ontable shot390)
  (ontable shot373)
  (dispenses dispenser403 ingredient256)
  (dispenses dispenser147 ingredient180)
  (dispenses dispenser362 ingredient19)
  (clean shaker16)
  (clean shot390)
  (clean shot373)
  (empty shaker16)
  (empty shot390)
  (empty shot373)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker16 l0)
  (shaker-level shaker16 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail337 ingredient180)
  (cocktail-part2 cocktail337 ingredient19)
)
 (:goal
  (and
      (contains shot390 cocktail337)
)))
