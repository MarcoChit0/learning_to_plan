(define (problem prob)
 (:domain barman)
 (:objects 
      shaker400 - shaker
      left right - hand
      shot94 shot262 - shot
      ingredient181 ingredient168 ingredient106 - ingredient
      cocktail307 - cocktail
      dispenser57 dispenser56 dispenser63 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker400)
  (ontable shot94)
  (ontable shot262)
  (dispenses dispenser57 ingredient181)
  (dispenses dispenser56 ingredient168)
  (dispenses dispenser63 ingredient106)
  (clean shaker400)
  (clean shot94)
  (clean shot262)
  (empty shaker400)
  (empty shot94)
  (empty shot262)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker400 l0)
  (shaker-level shaker400 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail307 ingredient106)
  (cocktail-part2 cocktail307 ingredient181)
)
 (:goal
  (and
      (contains shot94 cocktail307)
)))
