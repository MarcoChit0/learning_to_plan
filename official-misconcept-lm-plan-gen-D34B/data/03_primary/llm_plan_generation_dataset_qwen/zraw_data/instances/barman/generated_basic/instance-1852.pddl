(define (problem prob)
 (:domain barman)
 (:objects 
      shaker59 - shaker
      left right - hand
      shot418 shot172 - shot
      ingredient69 ingredient117 ingredient108 - ingredient
      cocktail317 - cocktail
      dispenser25 dispenser82 dispenser13 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker59)
  (ontable shot418)
  (ontable shot172)
  (dispenses dispenser25 ingredient69)
  (dispenses dispenser82 ingredient117)
  (dispenses dispenser13 ingredient108)
  (clean shaker59)
  (clean shot418)
  (clean shot172)
  (empty shaker59)
  (empty shot418)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker59 l0)
  (shaker-level shaker59 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail317 ingredient117)
  (cocktail-part2 cocktail317 ingredient108)
)
 (:goal
  (and
      (contains shot418 cocktail317)
)))
