(define (problem prob)
 (:domain barman)
 (:objects 
      shaker127 - shaker
      left right - hand
      shot34 shot71 shot161 shot405 - shot
      ingredient202 ingredient128 - ingredient
      cocktail66 - cocktail
      dispenser374 dispenser234 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker127)
  (ontable shot34)
  (ontable shot71)
  (ontable shot161)
  (ontable shot405)
  (dispenses dispenser374 ingredient202)
  (dispenses dispenser234 ingredient128)
  (clean shaker127)
  (clean shot34)
  (clean shot71)
  (clean shot161)
  (clean shot405)
  (empty shaker127)
  (empty shot34)
  (empty shot71)
  (empty shot161)
  (empty shot405)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker127 l0)
  (shaker-level shaker127 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail66 ingredient128)
  (cocktail-part2 cocktail66 ingredient202)
)
 (:goal
  (and
      (contains shot34 cocktail66)
      (contains shot71 cocktail66)
      (contains shot161 ingredient202)
)))
