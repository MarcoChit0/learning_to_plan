(define (problem prob)
 (:domain barman)
 (:objects 
      shaker216 - shaker
      left right - hand
      shot474 shot213 - shot
      ingredient328 ingredient72 ingredient299 - ingredient
      cocktail94 - cocktail
      dispenser79 dispenser441 dispenser308 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker216)
  (ontable shot474)
  (ontable shot213)
  (dispenses dispenser79 ingredient328)
  (dispenses dispenser441 ingredient72)
  (dispenses dispenser308 ingredient299)
  (clean shaker216)
  (clean shot474)
  (clean shot213)
  (empty shaker216)
  (empty shot474)
  (empty shot213)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker216 l0)
  (shaker-level shaker216 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail94 ingredient299)
  (cocktail-part2 cocktail94 ingredient72)
)
 (:goal
  (and
      (contains shot474 cocktail94)
)))
