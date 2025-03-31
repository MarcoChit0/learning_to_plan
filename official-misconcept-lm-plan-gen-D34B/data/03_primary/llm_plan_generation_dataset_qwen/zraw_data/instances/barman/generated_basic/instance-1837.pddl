(define (problem prob)
 (:domain barman)
 (:objects 
      shaker68 - shaker
      left right - hand
      shot400 shot383 - shot
      ingredient492 ingredient131 ingredient176 - ingredient
      cocktail3 - cocktail
      dispenser306 dispenser271 dispenser453 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker68)
  (ontable shot400)
  (ontable shot383)
  (dispenses dispenser306 ingredient492)
  (dispenses dispenser271 ingredient131)
  (dispenses dispenser453 ingredient176)
  (clean shaker68)
  (clean shot400)
  (clean shot383)
  (empty shaker68)
  (empty shot400)
  (empty shot383)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker68 l0)
  (shaker-level shaker68 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail3 ingredient131)
  (cocktail-part2 cocktail3 ingredient492)
)
 (:goal
  (and
      (contains shot400 cocktail3)
)))
