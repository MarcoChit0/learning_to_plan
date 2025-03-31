(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot459 shot327 shot334 - shot
      ingredient198 ingredient24 ingredient279 ingredient450 - ingredient
      cocktail1 - cocktail
      dispenser39 dispenser405 dispenser231 dispenser55 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot459)
  (ontable shot327)
  (ontable shot334)
  (dispenses dispenser39 ingredient198)
  (dispenses dispenser405 ingredient24)
  (dispenses dispenser231 ingredient279)
  (dispenses dispenser55 ingredient450)
  (clean shaker262)
  (clean shot459)
  (clean shot327)
  (clean shot334)
  (empty shaker262)
  (empty shot459)
  (empty shot327)
  (empty shot334)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient450)
  (cocktail-part2 cocktail1 ingredient24)
)
 (:goal
  (and
      (contains shot459 cocktail1)
      (contains shot327 cocktail1)
)))
