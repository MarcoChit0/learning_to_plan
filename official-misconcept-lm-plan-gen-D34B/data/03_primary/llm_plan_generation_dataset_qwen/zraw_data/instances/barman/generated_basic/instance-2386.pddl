(define (problem prob)
 (:domain barman)
 (:objects 
      shaker271 - shaker
      left right - hand
      shot471 shot284 - shot
      ingredient126 ingredient250 ingredient421 - ingredient
      cocktail90 - cocktail
      dispenser330 dispenser156 dispenser109 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker271)
  (ontable shot471)
  (ontable shot284)
  (dispenses dispenser330 ingredient126)
  (dispenses dispenser156 ingredient250)
  (dispenses dispenser109 ingredient421)
  (clean shaker271)
  (clean shot471)
  (clean shot284)
  (empty shaker271)
  (empty shot471)
  (empty shot284)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker271 l0)
  (shaker-level shaker271 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail90 ingredient126)
  (cocktail-part2 cocktail90 ingredient250)
)
 (:goal
  (and
      (contains shot471 cocktail90)
)))
