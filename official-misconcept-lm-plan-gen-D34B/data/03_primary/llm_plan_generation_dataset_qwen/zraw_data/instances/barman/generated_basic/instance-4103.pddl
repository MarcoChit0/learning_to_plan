(define (problem prob)
 (:domain barman)
 (:objects 
      shaker187 - shaker
      left right - hand
      shot222 shot176 - shot
      ingredient439 ingredient432 - ingredient
      cocktail215 - cocktail
      dispenser498 dispenser172 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker187)
  (ontable shot222)
  (ontable shot176)
  (dispenses dispenser498 ingredient439)
  (dispenses dispenser172 ingredient432)
  (clean shaker187)
  (clean shot222)
  (clean shot176)
  (empty shaker187)
  (empty shot222)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker187 l0)
  (shaker-level shaker187 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail215 ingredient439)
  (cocktail-part2 cocktail215 ingredient432)
)
 (:goal
  (and
      (contains shot222 cocktail215)
)))
