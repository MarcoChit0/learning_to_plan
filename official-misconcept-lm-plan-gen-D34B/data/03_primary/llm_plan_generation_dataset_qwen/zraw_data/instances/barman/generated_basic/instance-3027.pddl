(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot171 - shot
      ingredient372 ingredient426 ingredient287 - ingredient
      cocktail7 - cocktail
      dispenser451 dispenser12 dispenser252 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot171)
  (dispenses dispenser451 ingredient372)
  (dispenses dispenser12 ingredient426)
  (dispenses dispenser252 ingredient287)
  (clean shaker156)
  (clean shot171)
  (empty shaker156)
  (empty shot171)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail7 ingredient372)
  (cocktail-part2 cocktail7 ingredient426)
)
 (:goal
  (and
      (contains shot171 cocktail7)
)))
