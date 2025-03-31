(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot449 shot345 shot216 - shot
      ingredient439 ingredient78 ingredient234 ingredient430 - ingredient
      cocktail1 - cocktail
      dispenser46 dispenser82 dispenser120 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot449)
  (ontable shot345)
  (ontable shot216)
  (dispenses dispenser46 ingredient439)
  (dispenses dispenser82 ingredient78)
  (dispenses dispenser120 ingredient234)
  (dispenses dispenser32 ingredient430)
  (clean shaker99)
  (clean shot449)
  (clean shot345)
  (clean shot216)
  (empty shaker99)
  (empty shot449)
  (empty shot345)
  (empty shot216)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient430)
  (cocktail-part2 cocktail1 ingredient439)
)
 (:goal
  (and
      (contains shot449 cocktail1)
      (contains shot345 cocktail1)
)))
