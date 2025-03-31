(define (problem prob)
 (:domain barman)
 (:objects 
      shaker46 - shaker
      left right - hand
      shot82 - shot
      ingredient266 ingredient459 - ingredient
      cocktail3 - cocktail
      dispenser64 dispenser408 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker46)
  (ontable shot82)
  (dispenses dispenser64 ingredient266)
  (dispenses dispenser408 ingredient459)
  (clean shaker46)
  (clean shot82)
  (empty shaker46)
  (empty shot82)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker46 l0)
  (shaker-level shaker46 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail3 ingredient266)
  (cocktail-part2 cocktail3 ingredient459)
)
 (:goal
  (and
      (contains shot82 cocktail3)
)))
