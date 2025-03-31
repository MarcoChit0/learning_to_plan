(define (problem prob)
 (:domain barman)
 (:objects 
      shaker131 - shaker
      left right - hand
      shot165 shot95 - shot
      ingredient166 ingredient121 ingredient148 - ingredient
      cocktail1 - cocktail
      dispenser330 dispenser176 dispenser442 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker131)
  (ontable shot165)
  (ontable shot95)
  (dispenses dispenser330 ingredient166)
  (dispenses dispenser176 ingredient121)
  (dispenses dispenser442 ingredient148)
  (clean shaker131)
  (clean shot165)
  (clean shot95)
  (empty shaker131)
  (empty shot165)
  (empty shot95)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker131 l0)
  (shaker-level shaker131 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient121)
  (cocktail-part2 cocktail1 ingredient166)
)
 (:goal
  (and
      (contains shot165 cocktail1)
)))
