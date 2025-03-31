(define (problem prob)
 (:domain barman)
 (:objects 
      shaker311 - shaker
      left right - hand
      shot192 - shot
      ingredient4 ingredient153 - ingredient
      cocktail202 - cocktail
      dispenser444 dispenser333 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker311)
  (ontable shot192)
  (dispenses dispenser444 ingredient4)
  (dispenses dispenser333 ingredient153)
  (clean shaker311)
  (clean shot192)
  (empty shaker311)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker311 l0)
  (shaker-level shaker311 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail202 ingredient153)
  (cocktail-part2 cocktail202 ingredient4)
)
 (:goal
  (and
      (contains shot192 cocktail202)
)))
