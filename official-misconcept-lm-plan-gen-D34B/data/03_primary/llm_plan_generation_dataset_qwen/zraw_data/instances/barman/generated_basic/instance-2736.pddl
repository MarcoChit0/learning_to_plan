(define (problem prob)
 (:domain barman)
 (:objects 
      shaker250 - shaker
      left right - hand
      shot331 - shot
      ingredient345 ingredient394 - ingredient
      cocktail226 - cocktail
      dispenser343 dispenser27 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker250)
  (ontable shot331)
  (dispenses dispenser343 ingredient345)
  (dispenses dispenser27 ingredient394)
  (clean shaker250)
  (clean shot331)
  (empty shaker250)
  (empty shot331)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker250 l0)
  (shaker-level shaker250 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail226 ingredient345)
  (cocktail-part2 cocktail226 ingredient394)
)
 (:goal
  (and
      (contains shot331 cocktail226)
)))
