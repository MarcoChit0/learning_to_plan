(define (problem prob)
 (:domain barman)
 (:objects 
      shaker484 - shaker
      left right - hand
      shot471 shot270 - shot
      ingredient352 ingredient242 ingredient52 ingredient5 - ingredient
      cocktail48 - cocktail
      dispenser290 dispenser48 dispenser205 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker484)
  (ontable shot471)
  (ontable shot270)
  (dispenses dispenser290 ingredient352)
  (dispenses dispenser48 ingredient242)
  (dispenses dispenser205 ingredient52)
  (dispenses dispenser214 ingredient5)
  (clean shaker484)
  (clean shot471)
  (clean shot270)
  (empty shaker484)
  (empty shot471)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker484 l0)
  (shaker-level shaker484 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail48 ingredient5)
  (cocktail-part2 cocktail48 ingredient52)
)
 (:goal
  (and
      (contains shot471 cocktail48)
)))
