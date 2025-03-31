(define (problem prob)
 (:domain barman)
 (:objects 
      shaker379 - shaker
      left right - hand
      shot39 - shot
      ingredient222 ingredient72 - ingredient
      cocktail1 - cocktail
      dispenser223 dispenser167 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker379)
  (ontable shot39)
  (dispenses dispenser223 ingredient222)
  (dispenses dispenser167 ingredient72)
  (clean shaker379)
  (clean shot39)
  (empty shaker379)
  (empty shot39)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker379 l0)
  (shaker-level shaker379 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient222)
  (cocktail-part2 cocktail1 ingredient72)
)
 (:goal
  (and
      (contains shot39 cocktail1)
)))
