(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot31 - shot
      ingredient436 ingredient411 ingredient14 - ingredient
      cocktail165 - cocktail
      dispenser450 dispenser25 dispenser464 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot31)
  (dispenses dispenser450 ingredient436)
  (dispenses dispenser25 ingredient411)
  (dispenses dispenser464 ingredient14)
  (clean shaker179)
  (clean shot31)
  (empty shaker179)
  (empty shot31)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail165 ingredient14)
  (cocktail-part2 cocktail165 ingredient436)
)
 (:goal
  (and
      (contains shot31 cocktail165)
)))
