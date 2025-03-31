(define (problem prob)
 (:domain barman)
 (:objects 
      shaker308 - shaker
      left right - hand
      shot32 - shot
      ingredient16 ingredient190 - ingredient
      cocktail391 - cocktail
      dispenser109 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker308)
  (ontable shot32)
  (dispenses dispenser109 ingredient16)
  (dispenses dispenser21 ingredient190)
  (clean shaker308)
  (clean shot32)
  (empty shaker308)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker308 l0)
  (shaker-level shaker308 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail391 ingredient16)
  (cocktail-part2 cocktail391 ingredient190)
)
 (:goal
  (and
      (contains shot32 cocktail391)
)))
