(define (problem prob)
 (:domain barman)
 (:objects 
      shaker491 - shaker
      left right - hand
      shot326 shot215 shot334 - shot
      ingredient303 ingredient378 - ingredient
      cocktail200 - cocktail
      dispenser397 dispenser498 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker491)
  (ontable shot326)
  (ontable shot215)
  (ontable shot334)
  (dispenses dispenser397 ingredient303)
  (dispenses dispenser498 ingredient378)
  (clean shaker491)
  (clean shot326)
  (clean shot215)
  (clean shot334)
  (empty shaker491)
  (empty shot326)
  (empty shot215)
  (empty shot334)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker491 l0)
  (shaker-level shaker491 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail200 ingredient378)
  (cocktail-part2 cocktail200 ingredient303)
)
 (:goal
  (and
      (contains shot326 cocktail200)
      (contains shot215 cocktail200)
)))
