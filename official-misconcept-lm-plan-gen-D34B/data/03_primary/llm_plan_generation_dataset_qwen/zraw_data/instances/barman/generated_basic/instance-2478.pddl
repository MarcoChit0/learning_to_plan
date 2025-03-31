(define (problem prob)
 (:domain barman)
 (:objects 
      shaker29 - shaker
      left right - hand
      shot18 - shot
      ingredient60 ingredient4 ingredient411 - ingredient
      cocktail488 - cocktail
      dispenser332 dispenser310 dispenser212 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker29)
  (ontable shot18)
  (dispenses dispenser332 ingredient60)
  (dispenses dispenser310 ingredient4)
  (dispenses dispenser212 ingredient411)
  (clean shaker29)
  (clean shot18)
  (empty shaker29)
  (empty shot18)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker29 l0)
  (shaker-level shaker29 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail488 ingredient60)
  (cocktail-part2 cocktail488 ingredient4)
)
 (:goal
  (and
      (contains shot18 cocktail488)
)))
