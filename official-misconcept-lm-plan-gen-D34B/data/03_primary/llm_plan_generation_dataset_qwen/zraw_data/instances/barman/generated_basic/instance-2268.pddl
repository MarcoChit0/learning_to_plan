(define (problem prob)
 (:domain barman)
 (:objects 
      shaker475 - shaker
      left right - hand
      shot215 shot34 - shot
      ingredient376 ingredient402 ingredient198 ingredient13 - ingredient
      cocktail461 - cocktail
      dispenser12 dispenser42 dispenser202 dispenser15 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker475)
  (ontable shot215)
  (ontable shot34)
  (dispenses dispenser12 ingredient376)
  (dispenses dispenser42 ingredient402)
  (dispenses dispenser202 ingredient198)
  (dispenses dispenser15 ingredient13)
  (clean shaker475)
  (clean shot215)
  (clean shot34)
  (empty shaker475)
  (empty shot215)
  (empty shot34)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker475 l0)
  (shaker-level shaker475 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail461 ingredient376)
  (cocktail-part2 cocktail461 ingredient402)
)
 (:goal
  (and
      (contains shot215 cocktail461)
)))
