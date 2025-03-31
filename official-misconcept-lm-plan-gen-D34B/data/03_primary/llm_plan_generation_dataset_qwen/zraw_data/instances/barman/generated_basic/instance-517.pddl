(define (problem prob)
 (:domain barman)
 (:objects 
      shaker321 - shaker
      left right - hand
      shot153 shot233 - shot
      ingredient325 ingredient389 - ingredient
      cocktail1 - cocktail
      dispenser130 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker321)
  (ontable shot153)
  (ontable shot233)
  (dispenses dispenser130 ingredient325)
  (dispenses dispenser274 ingredient389)
  (clean shaker321)
  (clean shot153)
  (clean shot233)
  (empty shaker321)
  (empty shot153)
  (empty shot233)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker321 l0)
  (shaker-level shaker321 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient389)
  (cocktail-part2 cocktail1 ingredient325)
)
 (:goal
  (and
      (contains shot153 cocktail1)
)))
