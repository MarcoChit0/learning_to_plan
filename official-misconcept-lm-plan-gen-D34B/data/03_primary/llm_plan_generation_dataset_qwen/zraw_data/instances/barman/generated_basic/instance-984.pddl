(define (problem prob)
 (:domain barman)
 (:objects 
      shaker471 - shaker
      left right - hand
      shot24 shot38 - shot
      ingredient241 ingredient488 ingredient451 ingredient348 - ingredient
      cocktail1 - cocktail
      dispenser148 dispenser92 dispenser86 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker471)
  (ontable shot24)
  (ontable shot38)
  (dispenses dispenser148 ingredient241)
  (dispenses dispenser92 ingredient488)
  (dispenses dispenser86 ingredient451)
  (dispenses dispenser217 ingredient348)
  (clean shaker471)
  (clean shot24)
  (clean shot38)
  (empty shaker471)
  (empty shot24)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker471 l0)
  (shaker-level shaker471 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient451)
  (cocktail-part2 cocktail1 ingredient241)
)
 (:goal
  (and
      (contains shot24 cocktail1)
)))
