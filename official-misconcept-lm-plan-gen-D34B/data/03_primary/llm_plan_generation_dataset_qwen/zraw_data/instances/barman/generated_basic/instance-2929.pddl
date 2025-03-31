(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot73 - shot
      ingredient214 ingredient247 - ingredient
      cocktail19 - cocktail
      dispenser331 dispenser93 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot73)
  (dispenses dispenser331 ingredient214)
  (dispenses dispenser93 ingredient247)
  (clean shaker34)
  (clean shot73)
  (empty shaker34)
  (empty shot73)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail19 ingredient214)
  (cocktail-part2 cocktail19 ingredient247)
)
 (:goal
  (and
      (contains shot73 cocktail19)
)))
