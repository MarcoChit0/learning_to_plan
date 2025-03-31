(define (problem prob)
 (:domain barman)
 (:objects 
      shaker17 - shaker
      left right - hand
      shot113 shot20 - shot
      ingredient451 ingredient402 ingredient400 ingredient494 - ingredient
      cocktail5 - cocktail
      dispenser72 dispenser292 dispenser116 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker17)
  (ontable shot113)
  (ontable shot20)
  (dispenses dispenser72 ingredient451)
  (dispenses dispenser292 ingredient402)
  (dispenses dispenser116 ingredient400)
  (dispenses dispenser191 ingredient494)
  (clean shaker17)
  (clean shot113)
  (clean shot20)
  (empty shaker17)
  (empty shot113)
  (empty shot20)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker17 l0)
  (shaker-level shaker17 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail5 ingredient451)
  (cocktail-part2 cocktail5 ingredient402)
)
 (:goal
  (and
      (contains shot113 cocktail5)
)))
