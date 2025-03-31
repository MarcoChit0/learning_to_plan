(define (problem prob)
 (:domain barman)
 (:objects 
      shaker493 - shaker
      left right - hand
      shot123 shot498 - shot
      ingredient206 ingredient129 - ingredient
      cocktail1 - cocktail
      dispenser460 dispenser120 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker493)
  (ontable shot123)
  (ontable shot498)
  (dispenses dispenser460 ingredient206)
  (dispenses dispenser120 ingredient129)
  (clean shaker493)
  (clean shot123)
  (clean shot498)
  (empty shaker493)
  (empty shot123)
  (empty shot498)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker493 l0)
  (shaker-level shaker493 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient206)
  (cocktail-part2 cocktail1 ingredient129)
)
 (:goal
  (and
      (contains shot123 cocktail1)
)))
