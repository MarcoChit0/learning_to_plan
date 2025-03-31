(define (problem prob)
 (:domain barman)
 (:objects 
      shaker274 - shaker
      left right - hand
      shot80 shot95 shot248 - shot
      ingredient341 ingredient108 ingredient64 - ingredient
      cocktail1 - cocktail
      dispenser0 dispenser100 dispenser415 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker274)
  (ontable shot80)
  (ontable shot95)
  (ontable shot248)
  (dispenses dispenser0 ingredient341)
  (dispenses dispenser100 ingredient108)
  (dispenses dispenser415 ingredient64)
  (clean shaker274)
  (clean shot80)
  (clean shot95)
  (clean shot248)
  (empty shaker274)
  (empty shot80)
  (empty shot95)
  (empty shot248)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker274 l0)
  (shaker-level shaker274 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient108)
  (cocktail-part2 cocktail1 ingredient341)
)
 (:goal
  (and
      (contains shot80 cocktail1)
      (contains shot95 ingredient64)
)))
