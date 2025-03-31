(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot317 - shot
      ingredient189 ingredient422 ingredient193 - ingredient
      cocktail313 - cocktail
      dispenser398 dispenser40 dispenser269 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot317)
  (dispenses dispenser398 ingredient189)
  (dispenses dispenser40 ingredient422)
  (dispenses dispenser269 ingredient193)
  (clean shaker356)
  (clean shot317)
  (empty shaker356)
  (empty shot317)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail313 ingredient422)
  (cocktail-part2 cocktail313 ingredient193)
)
 (:goal
  (and
      (contains shot317 cocktail313)
)))
