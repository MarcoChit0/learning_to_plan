(define (problem prob)
 (:domain barman)
 (:objects 
      shaker235 - shaker
      left right - hand
      shot132 - shot
      ingredient241 ingredient322 - ingredient
      cocktail85 - cocktail
      dispenser349 dispenser277 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker235)
  (ontable shot132)
  (dispenses dispenser349 ingredient241)
  (dispenses dispenser277 ingredient322)
  (clean shaker235)
  (clean shot132)
  (empty shaker235)
  (empty shot132)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker235 l0)
  (shaker-level shaker235 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail85 ingredient241)
  (cocktail-part2 cocktail85 ingredient322)
)
 (:goal
  (and
      (contains shot132 cocktail85)
)))
