(define (problem prob)
 (:domain barman)
 (:objects 
      shaker330 - shaker
      left right - hand
      shot490 shot415 - shot
      ingredient75 ingredient19 - ingredient
      cocktail119 - cocktail
      dispenser326 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker330)
  (ontable shot490)
  (ontable shot415)
  (dispenses dispenser326 ingredient75)
  (dispenses dispenser283 ingredient19)
  (clean shaker330)
  (clean shot490)
  (clean shot415)
  (empty shaker330)
  (empty shot490)
  (empty shot415)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker330 l0)
  (shaker-level shaker330 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail119 ingredient19)
  (cocktail-part2 cocktail119 ingredient75)
)
 (:goal
  (and
      (contains shot490 cocktail119)
)))
