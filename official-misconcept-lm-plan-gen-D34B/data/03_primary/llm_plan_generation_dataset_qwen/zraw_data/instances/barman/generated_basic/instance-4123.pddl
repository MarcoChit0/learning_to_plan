(define (problem prob)
 (:domain barman)
 (:objects 
      shaker239 - shaker
      left right - hand
      shot402 shot451 - shot
      ingredient228 ingredient473 ingredient450 - ingredient
      cocktail63 - cocktail
      dispenser37 dispenser4 dispenser261 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker239)
  (ontable shot402)
  (ontable shot451)
  (dispenses dispenser37 ingredient228)
  (dispenses dispenser4 ingredient473)
  (dispenses dispenser261 ingredient450)
  (clean shaker239)
  (clean shot402)
  (clean shot451)
  (empty shaker239)
  (empty shot402)
  (empty shot451)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker239 l0)
  (shaker-level shaker239 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail63 ingredient228)
  (cocktail-part2 cocktail63 ingredient473)
)
 (:goal
  (and
      (contains shot402 cocktail63)
)))
