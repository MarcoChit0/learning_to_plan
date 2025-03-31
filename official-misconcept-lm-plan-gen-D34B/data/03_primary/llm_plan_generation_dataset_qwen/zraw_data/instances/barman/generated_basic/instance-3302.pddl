(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot496 - shot
      ingredient335 ingredient364 - ingredient
      cocktail363 - cocktail
      dispenser54 dispenser155 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot496)
  (dispenses dispenser54 ingredient335)
  (dispenses dispenser155 ingredient364)
  (clean shaker15)
  (clean shot496)
  (empty shaker15)
  (empty shot496)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail363 ingredient364)
  (cocktail-part2 cocktail363 ingredient335)
)
 (:goal
  (and
      (contains shot496 cocktail363)
)))
