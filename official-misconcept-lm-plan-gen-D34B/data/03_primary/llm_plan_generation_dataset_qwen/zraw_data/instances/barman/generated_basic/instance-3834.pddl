(define (problem prob)
 (:domain barman)
 (:objects 
      shaker70 - shaker
      left right - hand
      shot494 shot340 shot388 - shot
      ingredient344 ingredient376 - ingredient
      cocktail401 - cocktail
      dispenser306 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker70)
  (ontable shot494)
  (ontable shot340)
  (ontable shot388)
  (dispenses dispenser306 ingredient344)
  (dispenses dispenser207 ingredient376)
  (clean shaker70)
  (clean shot494)
  (clean shot340)
  (clean shot388)
  (empty shaker70)
  (empty shot494)
  (empty shot340)
  (empty shot388)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker70 l0)
  (shaker-level shaker70 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail401 ingredient376)
  (cocktail-part2 cocktail401 ingredient344)
)
 (:goal
  (and
      (contains shot494 cocktail401)
      (contains shot340 ingredient344)
)))
