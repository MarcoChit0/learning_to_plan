(define (problem prob)
 (:domain barman)
 (:objects 
      shaker422 - shaker
      left right - hand
      shot403 - shot
      ingredient246 ingredient498 - ingredient
      cocktail114 - cocktail
      dispenser238 dispenser151 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker422)
  (ontable shot403)
  (dispenses dispenser238 ingredient246)
  (dispenses dispenser151 ingredient498)
  (clean shaker422)
  (clean shot403)
  (empty shaker422)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker422 l0)
  (shaker-level shaker422 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail114 ingredient246)
  (cocktail-part2 cocktail114 ingredient498)
)
 (:goal
  (and
      (contains shot403 cocktail114)
)))
