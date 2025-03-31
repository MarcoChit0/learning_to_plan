(define (problem prob)
 (:domain barman)
 (:objects 
      shaker285 - shaker
      left right - hand
      shot205 shot202 shot142 - shot
      ingredient465 ingredient93 ingredient386 ingredient429 - ingredient
      cocktail1 - cocktail
      dispenser353 dispenser472 dispenser319 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker285)
  (ontable shot205)
  (ontable shot202)
  (ontable shot142)
  (dispenses dispenser353 ingredient465)
  (dispenses dispenser472 ingredient93)
  (dispenses dispenser319 ingredient386)
  (dispenses dispenser224 ingredient429)
  (clean shaker285)
  (clean shot205)
  (clean shot202)
  (clean shot142)
  (empty shaker285)
  (empty shot205)
  (empty shot202)
  (empty shot142)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker285 l0)
  (shaker-level shaker285 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient465)
  (cocktail-part2 cocktail1 ingredient386)
)
 (:goal
  (and
      (contains shot205 cocktail1)
      (contains shot202 cocktail1)
)))
