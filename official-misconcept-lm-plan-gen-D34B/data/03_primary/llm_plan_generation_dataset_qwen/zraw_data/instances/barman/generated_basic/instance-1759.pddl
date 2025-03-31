(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot357 shot61 shot237 - shot
      ingredient324 ingredient107 - ingredient
      cocktail13 - cocktail
      dispenser113 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot357)
  (ontable shot61)
  (ontable shot237)
  (dispenses dispenser113 ingredient324)
  (dispenses dispenser458 ingredient107)
  (clean shaker34)
  (clean shot357)
  (clean shot61)
  (clean shot237)
  (empty shaker34)
  (empty shot357)
  (empty shot61)
  (empty shot237)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail13 ingredient324)
  (cocktail-part2 cocktail13 ingredient107)
)
 (:goal
  (and
      (contains shot357 cocktail13)
      (contains shot61 cocktail13)
)))
