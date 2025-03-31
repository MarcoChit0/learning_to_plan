(define (problem prob)
 (:domain barman)
 (:objects 
      shaker471 - shaker
      left right - hand
      shot454 - shot
      ingredient241 ingredient135 - ingredient
      cocktail1 - cocktail
      dispenser195 dispenser412 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker471)
  (ontable shot454)
  (dispenses dispenser195 ingredient241)
  (dispenses dispenser412 ingredient135)
  (clean shaker471)
  (clean shot454)
  (empty shaker471)
  (empty shot454)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker471 l0)
  (shaker-level shaker471 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient135)
  (cocktail-part2 cocktail1 ingredient241)
)
 (:goal
  (and
      (contains shot454 cocktail1)
)))
