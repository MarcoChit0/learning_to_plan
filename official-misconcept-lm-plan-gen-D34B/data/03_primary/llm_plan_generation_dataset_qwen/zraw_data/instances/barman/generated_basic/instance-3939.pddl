(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot106 shot87 - shot
      ingredient345 ingredient206 - ingredient
      cocktail184 - cocktail
      dispenser437 dispenser285 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot106)
  (ontable shot87)
  (dispenses dispenser437 ingredient345)
  (dispenses dispenser285 ingredient206)
  (clean shaker136)
  (clean shot106)
  (clean shot87)
  (empty shaker136)
  (empty shot106)
  (empty shot87)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail184 ingredient345)
  (cocktail-part2 cocktail184 ingredient206)
)
 (:goal
  (and
      (contains shot106 cocktail184)
)))
