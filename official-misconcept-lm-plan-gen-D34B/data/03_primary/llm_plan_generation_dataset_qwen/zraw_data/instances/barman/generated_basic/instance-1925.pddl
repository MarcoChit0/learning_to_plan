(define (problem prob)
 (:domain barman)
 (:objects 
      shaker353 - shaker
      left right - hand
      shot306 - shot
      ingredient237 ingredient418 - ingredient
      cocktail458 - cocktail
      dispenser229 dispenser420 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker353)
  (ontable shot306)
  (dispenses dispenser229 ingredient237)
  (dispenses dispenser420 ingredient418)
  (clean shaker353)
  (clean shot306)
  (empty shaker353)
  (empty shot306)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker353 l0)
  (shaker-level shaker353 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail458 ingredient418)
  (cocktail-part2 cocktail458 ingredient237)
)
 (:goal
  (and
      (contains shot306 cocktail458)
)))
