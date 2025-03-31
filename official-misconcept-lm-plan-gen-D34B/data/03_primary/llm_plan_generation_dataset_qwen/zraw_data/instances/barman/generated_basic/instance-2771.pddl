(define (problem prob)
 (:domain barman)
 (:objects 
      shaker345 - shaker
      left right - hand
      shot426 shot377 - shot
      ingredient262 ingredient276 ingredient234 - ingredient
      cocktail12 - cocktail
      dispenser106 dispenser414 dispenser183 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker345)
  (ontable shot426)
  (ontable shot377)
  (dispenses dispenser106 ingredient262)
  (dispenses dispenser414 ingredient276)
  (dispenses dispenser183 ingredient234)
  (clean shaker345)
  (clean shot426)
  (clean shot377)
  (empty shaker345)
  (empty shot426)
  (empty shot377)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker345 l0)
  (shaker-level shaker345 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail12 ingredient276)
  (cocktail-part2 cocktail12 ingredient262)
)
 (:goal
  (and
      (contains shot426 cocktail12)
)))
