(define (problem prob)
 (:domain barman)
 (:objects 
      shaker95 - shaker
      left right - hand
      shot144 - shot
      ingredient323 ingredient247 - ingredient
      cocktail1 - cocktail
      dispenser137 dispenser463 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker95)
  (ontable shot144)
  (dispenses dispenser137 ingredient323)
  (dispenses dispenser463 ingredient247)
  (clean shaker95)
  (clean shot144)
  (empty shaker95)
  (empty shot144)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker95 l0)
  (shaker-level shaker95 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient323)
  (cocktail-part2 cocktail1 ingredient247)
)
 (:goal
  (and
      (contains shot144 cocktail1)
)))
