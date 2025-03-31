(define (problem prob)
 (:domain barman)
 (:objects 
      shaker357 - shaker
      left right - hand
      shot259 shot355 - shot
      ingredient267 ingredient416 ingredient114 ingredient248 - ingredient
      cocktail145 - cocktail
      dispenser63 dispenser52 dispenser336 dispenser269 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker357)
  (ontable shot259)
  (ontable shot355)
  (dispenses dispenser63 ingredient267)
  (dispenses dispenser52 ingredient416)
  (dispenses dispenser336 ingredient114)
  (dispenses dispenser269 ingredient248)
  (clean shaker357)
  (clean shot259)
  (clean shot355)
  (empty shaker357)
  (empty shot259)
  (empty shot355)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker357 l0)
  (shaker-level shaker357 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail145 ingredient416)
  (cocktail-part2 cocktail145 ingredient248)
)
 (:goal
  (and
      (contains shot259 cocktail145)
)))
