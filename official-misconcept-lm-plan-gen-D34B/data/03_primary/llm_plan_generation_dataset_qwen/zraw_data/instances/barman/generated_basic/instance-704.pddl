(define (problem prob)
 (:domain barman)
 (:objects 
      shaker334 - shaker
      left right - hand
      shot349 shot68 - shot
      ingredient299 ingredient306 ingredient358 - ingredient
      cocktail1 - cocktail
      dispenser413 dispenser372 dispenser272 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker334)
  (ontable shot349)
  (ontable shot68)
  (dispenses dispenser413 ingredient299)
  (dispenses dispenser372 ingredient306)
  (dispenses dispenser272 ingredient358)
  (clean shaker334)
  (clean shot349)
  (clean shot68)
  (empty shaker334)
  (empty shot349)
  (empty shot68)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker334 l0)
  (shaker-level shaker334 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient358)
  (cocktail-part2 cocktail1 ingredient306)
)
 (:goal
  (and
      (contains shot349 cocktail1)
)))
