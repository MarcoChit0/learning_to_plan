(define (problem prob)
 (:domain barman)
 (:objects 
      shaker199 - shaker
      left right - hand
      shot415 shot213 - shot
      ingredient287 ingredient101 ingredient140 ingredient492 - ingredient
      cocktail1 - cocktail
      dispenser413 dispenser57 dispenser360 dispenser183 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker199)
  (ontable shot415)
  (ontable shot213)
  (dispenses dispenser413 ingredient287)
  (dispenses dispenser57 ingredient101)
  (dispenses dispenser360 ingredient140)
  (dispenses dispenser183 ingredient492)
  (clean shaker199)
  (clean shot415)
  (clean shot213)
  (empty shaker199)
  (empty shot415)
  (empty shot213)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker199 l0)
  (shaker-level shaker199 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient101)
  (cocktail-part2 cocktail1 ingredient140)
)
 (:goal
  (and
      (contains shot415 cocktail1)
)))
