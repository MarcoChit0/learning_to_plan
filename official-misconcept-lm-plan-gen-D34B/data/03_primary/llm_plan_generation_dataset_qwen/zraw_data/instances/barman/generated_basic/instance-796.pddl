(define (problem prob)
 (:domain barman)
 (:objects 
      shaker271 - shaker
      left right - hand
      shot26 - shot
      ingredient352 ingredient110 - ingredient
      cocktail1 - cocktail
      dispenser250 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker271)
  (ontable shot26)
  (dispenses dispenser250 ingredient352)
  (dispenses dispenser306 ingredient110)
  (clean shaker271)
  (clean shot26)
  (empty shaker271)
  (empty shot26)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker271 l0)
  (shaker-level shaker271 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient352)
  (cocktail-part2 cocktail1 ingredient110)
)
 (:goal
  (and
      (contains shot26 cocktail1)
)))
