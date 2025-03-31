(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot407 - shot
      ingredient262 ingredient150 - ingredient
      cocktail181 - cocktail
      dispenser116 dispenser17 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot407)
  (dispenses dispenser116 ingredient262)
  (dispenses dispenser17 ingredient150)
  (clean shaker234)
  (clean shot407)
  (empty shaker234)
  (empty shot407)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail181 ingredient150)
  (cocktail-part2 cocktail181 ingredient262)
)
 (:goal
  (and
      (contains shot407 cocktail181)
)))
