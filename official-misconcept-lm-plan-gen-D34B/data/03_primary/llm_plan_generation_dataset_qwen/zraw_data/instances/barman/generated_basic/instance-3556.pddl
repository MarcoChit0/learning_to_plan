(define (problem prob)
 (:domain barman)
 (:objects 
      shaker43 - shaker
      left right - hand
      shot304 shot229 - shot
      ingredient291 ingredient293 ingredient427 ingredient143 - ingredient
      cocktail168 - cocktail
      dispenser345 dispenser215 dispenser106 dispenser257 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker43)
  (ontable shot304)
  (ontable shot229)
  (dispenses dispenser345 ingredient291)
  (dispenses dispenser215 ingredient293)
  (dispenses dispenser106 ingredient427)
  (dispenses dispenser257 ingredient143)
  (clean shaker43)
  (clean shot304)
  (clean shot229)
  (empty shaker43)
  (empty shot304)
  (empty shot229)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker43 l0)
  (shaker-level shaker43 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail168 ingredient427)
  (cocktail-part2 cocktail168 ingredient143)
)
 (:goal
  (and
      (contains shot304 cocktail168)
)))
