(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot178 - shot
      ingredient152 ingredient96 ingredient131 ingredient94 - ingredient
      cocktail1 - cocktail
      dispenser398 dispenser246 dispenser496 dispenser171 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot178)
  (dispenses dispenser398 ingredient152)
  (dispenses dispenser246 ingredient96)
  (dispenses dispenser496 ingredient131)
  (dispenses dispenser171 ingredient94)
  (clean shaker449)
  (clean shot178)
  (empty shaker449)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient131)
  (cocktail-part2 cocktail1 ingredient152)
)
 (:goal
  (and
      (contains shot178 cocktail1)
)))
