(define (problem prob)
 (:domain barman)
 (:objects 
      shaker211 - shaker
      left right - hand
      shot229 shot119 - shot
      ingredient26 ingredient108 ingredient309 - ingredient
      cocktail250 - cocktail
      dispenser285 dispenser94 dispenser18 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker211)
  (ontable shot229)
  (ontable shot119)
  (dispenses dispenser285 ingredient26)
  (dispenses dispenser94 ingredient108)
  (dispenses dispenser18 ingredient309)
  (clean shaker211)
  (clean shot229)
  (clean shot119)
  (empty shaker211)
  (empty shot229)
  (empty shot119)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker211 l0)
  (shaker-level shaker211 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail250 ingredient26)
  (cocktail-part2 cocktail250 ingredient309)
)
 (:goal
  (and
      (contains shot229 cocktail250)
)))
