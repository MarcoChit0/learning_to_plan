(define (problem prob)
 (:domain barman)
 (:objects 
      shaker159 - shaker
      left right - hand
      shot321 shot423 - shot
      ingredient269 ingredient121 - ingredient
      cocktail1 - cocktail
      dispenser90 dispenser374 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker159)
  (ontable shot321)
  (ontable shot423)
  (dispenses dispenser90 ingredient269)
  (dispenses dispenser374 ingredient121)
  (clean shaker159)
  (clean shot321)
  (clean shot423)
  (empty shaker159)
  (empty shot321)
  (empty shot423)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker159 l0)
  (shaker-level shaker159 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient121)
  (cocktail-part2 cocktail1 ingredient269)
)
 (:goal
  (and
      (contains shot321 cocktail1)
)))
