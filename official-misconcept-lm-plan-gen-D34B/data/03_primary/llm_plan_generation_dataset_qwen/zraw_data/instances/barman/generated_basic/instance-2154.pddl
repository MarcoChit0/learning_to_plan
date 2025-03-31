(define (problem prob)
 (:domain barman)
 (:objects 
      shaker369 - shaker
      left right - hand
      shot219 - shot
      ingredient303 ingredient77 - ingredient
      cocktail101 - cocktail
      dispenser408 dispenser179 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker369)
  (ontable shot219)
  (dispenses dispenser408 ingredient303)
  (dispenses dispenser179 ingredient77)
  (clean shaker369)
  (clean shot219)
  (empty shaker369)
  (empty shot219)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker369 l0)
  (shaker-level shaker369 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail101 ingredient303)
  (cocktail-part2 cocktail101 ingredient77)
)
 (:goal
  (and
      (contains shot219 cocktail101)
)))
