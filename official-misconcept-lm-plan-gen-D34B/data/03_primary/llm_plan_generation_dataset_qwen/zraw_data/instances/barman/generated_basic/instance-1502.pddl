(define (problem prob)
 (:domain barman)
 (:objects 
      shaker495 - shaker
      left right - hand
      shot488 shot67 - shot
      ingredient471 ingredient470 ingredient85 ingredient388 - ingredient
      cocktail321 - cocktail
      dispenser354 dispenser389 dispenser84 dispenser330 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker495)
  (ontable shot488)
  (ontable shot67)
  (dispenses dispenser354 ingredient471)
  (dispenses dispenser389 ingredient470)
  (dispenses dispenser84 ingredient85)
  (dispenses dispenser330 ingredient388)
  (clean shaker495)
  (clean shot488)
  (clean shot67)
  (empty shaker495)
  (empty shot488)
  (empty shot67)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker495 l0)
  (shaker-level shaker495 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail321 ingredient470)
  (cocktail-part2 cocktail321 ingredient471)
)
 (:goal
  (and
      (contains shot488 cocktail321)
)))
