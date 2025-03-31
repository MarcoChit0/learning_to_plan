(define (problem prob)
 (:domain barman)
 (:objects 
      shaker353 - shaker
      left right - hand
      shot483 - shot
      ingredient199 ingredient317 ingredient194 - ingredient
      cocktail1 - cocktail
      dispenser271 dispenser146 dispenser7 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker353)
  (ontable shot483)
  (dispenses dispenser271 ingredient199)
  (dispenses dispenser146 ingredient317)
  (dispenses dispenser7 ingredient194)
  (clean shaker353)
  (clean shot483)
  (empty shaker353)
  (empty shot483)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker353 l0)
  (shaker-level shaker353 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient199)
  (cocktail-part2 cocktail1 ingredient317)
)
 (:goal
  (and
      (contains shot483 cocktail1)
)))
