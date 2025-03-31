(define (problem prob)
 (:domain barman)
 (:objects 
      shaker154 - shaker
      left right - hand
      shot282 shot168 - shot
      ingredient317 ingredient344 ingredient305 ingredient468 - ingredient
      cocktail387 - cocktail
      dispenser477 dispenser333 dispenser18 dispenser57 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker154)
  (ontable shot282)
  (ontable shot168)
  (dispenses dispenser477 ingredient317)
  (dispenses dispenser333 ingredient344)
  (dispenses dispenser18 ingredient305)
  (dispenses dispenser57 ingredient468)
  (clean shaker154)
  (clean shot282)
  (clean shot168)
  (empty shaker154)
  (empty shot282)
  (empty shot168)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker154 l0)
  (shaker-level shaker154 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail387 ingredient468)
  (cocktail-part2 cocktail387 ingredient344)
)
 (:goal
  (and
      (contains shot282 cocktail387)
)))
