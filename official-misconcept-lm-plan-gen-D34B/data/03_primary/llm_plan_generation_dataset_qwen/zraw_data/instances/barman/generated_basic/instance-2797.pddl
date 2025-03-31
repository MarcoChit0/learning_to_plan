(define (problem prob)
 (:domain barman)
 (:objects 
      shaker79 - shaker
      left right - hand
      shot482 shot21 - shot
      ingredient434 ingredient243 - ingredient
      cocktail273 - cocktail
      dispenser91 dispenser164 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker79)
  (ontable shot482)
  (ontable shot21)
  (dispenses dispenser91 ingredient434)
  (dispenses dispenser164 ingredient243)
  (clean shaker79)
  (clean shot482)
  (clean shot21)
  (empty shaker79)
  (empty shot482)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker79 l0)
  (shaker-level shaker79 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail273 ingredient434)
  (cocktail-part2 cocktail273 ingredient243)
)
 (:goal
  (and
      (contains shot482 cocktail273)
)))
