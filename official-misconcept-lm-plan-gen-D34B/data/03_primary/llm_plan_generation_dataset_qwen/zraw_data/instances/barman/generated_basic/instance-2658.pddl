(define (problem prob)
 (:domain barman)
 (:objects 
      shaker207 - shaker
      left right - hand
      shot242 - shot
      ingredient427 ingredient478 ingredient476 ingredient450 - ingredient
      cocktail471 - cocktail
      dispenser347 dispenser27 dispenser375 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker207)
  (ontable shot242)
  (dispenses dispenser347 ingredient427)
  (dispenses dispenser27 ingredient478)
  (dispenses dispenser375 ingredient476)
  (dispenses dispenser9 ingredient450)
  (clean shaker207)
  (clean shot242)
  (empty shaker207)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker207 l0)
  (shaker-level shaker207 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail471 ingredient427)
  (cocktail-part2 cocktail471 ingredient450)
)
 (:goal
  (and
      (contains shot242 cocktail471)
)))
