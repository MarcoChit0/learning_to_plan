(define (problem prob)
 (:domain barman)
 (:objects 
      shaker395 - shaker
      left right - hand
      shot343 shot47 shot246 - shot
      ingredient403 ingredient430 ingredient482 ingredient495 - ingredient
      cocktail1 - cocktail
      dispenser257 dispenser216 dispenser210 dispenser223 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker395)
  (ontable shot343)
  (ontable shot47)
  (ontable shot246)
  (dispenses dispenser257 ingredient403)
  (dispenses dispenser216 ingredient430)
  (dispenses dispenser210 ingredient482)
  (dispenses dispenser223 ingredient495)
  (clean shaker395)
  (clean shot343)
  (clean shot47)
  (clean shot246)
  (empty shaker395)
  (empty shot343)
  (empty shot47)
  (empty shot246)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker395 l0)
  (shaker-level shaker395 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient495)
  (cocktail-part2 cocktail1 ingredient482)
)
 (:goal
  (and
      (contains shot343 cocktail1)
      (contains shot47 cocktail1)
)))
