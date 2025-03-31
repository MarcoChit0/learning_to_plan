(define (problem prob)
 (:domain barman)
 (:objects 
      shaker340 - shaker
      left right - hand
      shot166 shot273 shot342 - shot
      ingredient243 ingredient13 ingredient157 ingredient30 - ingredient
      cocktail1 - cocktail
      dispenser310 dispenser325 dispenser276 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker340)
  (ontable shot166)
  (ontable shot273)
  (ontable shot342)
  (dispenses dispenser310 ingredient243)
  (dispenses dispenser325 ingredient13)
  (dispenses dispenser276 ingredient157)
  (dispenses dispenser397 ingredient30)
  (clean shaker340)
  (clean shot166)
  (clean shot273)
  (clean shot342)
  (empty shaker340)
  (empty shot166)
  (empty shot273)
  (empty shot342)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker340 l0)
  (shaker-level shaker340 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient243)
  (cocktail-part2 cocktail1 ingredient13)
)
 (:goal
  (and
      (contains shot166 cocktail1)
      (contains shot273 ingredient30)
)))
