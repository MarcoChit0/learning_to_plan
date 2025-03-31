(define (problem prob)
 (:domain barman)
 (:objects 
      shaker222 - shaker
      left right - hand
      shot142 shot136 - shot
      ingredient442 ingredient396 ingredient370 - ingredient
      cocktail1 - cocktail
      dispenser410 dispenser95 dispenser237 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker222)
  (ontable shot142)
  (ontable shot136)
  (dispenses dispenser410 ingredient442)
  (dispenses dispenser95 ingredient396)
  (dispenses dispenser237 ingredient370)
  (clean shaker222)
  (clean shot142)
  (clean shot136)
  (empty shaker222)
  (empty shot142)
  (empty shot136)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker222 l0)
  (shaker-level shaker222 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient370)
  (cocktail-part2 cocktail1 ingredient396)
)
 (:goal
  (and
      (contains shot142 cocktail1)
)))
