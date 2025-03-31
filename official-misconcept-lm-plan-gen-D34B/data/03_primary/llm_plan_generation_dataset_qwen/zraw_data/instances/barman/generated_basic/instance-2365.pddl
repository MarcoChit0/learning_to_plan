(define (problem prob)
 (:domain barman)
 (:objects 
      shaker252 - shaker
      left right - hand
      shot83 - shot
      ingredient482 ingredient472 ingredient170 ingredient124 - ingredient
      cocktail487 - cocktail
      dispenser364 dispenser490 dispenser347 dispenser244 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker252)
  (ontable shot83)
  (dispenses dispenser364 ingredient482)
  (dispenses dispenser490 ingredient472)
  (dispenses dispenser347 ingredient170)
  (dispenses dispenser244 ingredient124)
  (clean shaker252)
  (clean shot83)
  (empty shaker252)
  (empty shot83)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker252 l0)
  (shaker-level shaker252 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail487 ingredient124)
  (cocktail-part2 cocktail487 ingredient170)
)
 (:goal
  (and
      (contains shot83 cocktail487)
)))
