(define (problem prob)
 (:domain barman)
 (:objects 
      shaker425 - shaker
      left right - hand
      shot212 - shot
      ingredient262 ingredient108 ingredient19 - ingredient
      cocktail431 - cocktail
      dispenser237 dispenser283 dispenser221 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker425)
  (ontable shot212)
  (dispenses dispenser237 ingredient262)
  (dispenses dispenser283 ingredient108)
  (dispenses dispenser221 ingredient19)
  (clean shaker425)
  (clean shot212)
  (empty shaker425)
  (empty shot212)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker425 l0)
  (shaker-level shaker425 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail431 ingredient108)
  (cocktail-part2 cocktail431 ingredient262)
)
 (:goal
  (and
      (contains shot212 cocktail431)
)))
