(define (problem prob)
 (:domain barman)
 (:objects 
      shaker118 - shaker
      left right - hand
      shot372 - shot
      ingredient411 ingredient455 - ingredient
      cocktail415 - cocktail
      dispenser49 dispenser430 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker118)
  (ontable shot372)
  (dispenses dispenser49 ingredient411)
  (dispenses dispenser430 ingredient455)
  (clean shaker118)
  (clean shot372)
  (empty shaker118)
  (empty shot372)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker118 l0)
  (shaker-level shaker118 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail415 ingredient455)
  (cocktail-part2 cocktail415 ingredient411)
)
 (:goal
  (and
      (contains shot372 cocktail415)
)))
