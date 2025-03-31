(define (problem prob)
 (:domain barman)
 (:objects 
      shaker322 - shaker
      left right - hand
      shot99 - shot
      ingredient21 ingredient345 - ingredient
      cocktail358 - cocktail
      dispenser309 dispenser124 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker322)
  (ontable shot99)
  (dispenses dispenser309 ingredient21)
  (dispenses dispenser124 ingredient345)
  (clean shaker322)
  (clean shot99)
  (empty shaker322)
  (empty shot99)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker322 l0)
  (shaker-level shaker322 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail358 ingredient21)
  (cocktail-part2 cocktail358 ingredient345)
)
 (:goal
  (and
      (contains shot99 cocktail358)
)))
