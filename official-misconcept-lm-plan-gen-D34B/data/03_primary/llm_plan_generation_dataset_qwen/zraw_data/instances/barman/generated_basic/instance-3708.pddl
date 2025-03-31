(define (problem prob)
 (:domain barman)
 (:objects 
      shaker292 - shaker
      left right - hand
      shot490 shot495 - shot
      ingredient193 ingredient6 ingredient296 - ingredient
      cocktail149 - cocktail
      dispenser58 dispenser282 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker292)
  (ontable shot490)
  (ontable shot495)
  (dispenses dispenser58 ingredient193)
  (dispenses dispenser282 ingredient6)
  (dispenses dispenser290 ingredient296)
  (clean shaker292)
  (clean shot490)
  (clean shot495)
  (empty shaker292)
  (empty shot490)
  (empty shot495)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker292 l0)
  (shaker-level shaker292 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail149 ingredient296)
  (cocktail-part2 cocktail149 ingredient193)
)
 (:goal
  (and
      (contains shot490 cocktail149)
)))
