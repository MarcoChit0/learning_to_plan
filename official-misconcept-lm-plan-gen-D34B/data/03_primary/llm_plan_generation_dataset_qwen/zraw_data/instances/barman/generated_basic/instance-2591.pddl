(define (problem prob)
 (:domain barman)
 (:objects 
      shaker354 - shaker
      left right - hand
      shot180 shot94 - shot
      ingredient150 ingredient379 ingredient114 - ingredient
      cocktail34 - cocktail
      dispenser489 dispenser459 dispenser322 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker354)
  (ontable shot180)
  (ontable shot94)
  (dispenses dispenser489 ingredient150)
  (dispenses dispenser459 ingredient379)
  (dispenses dispenser322 ingredient114)
  (clean shaker354)
  (clean shot180)
  (clean shot94)
  (empty shaker354)
  (empty shot180)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker354 l0)
  (shaker-level shaker354 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail34 ingredient114)
  (cocktail-part2 cocktail34 ingredient379)
)
 (:goal
  (and
      (contains shot180 cocktail34)
)))
