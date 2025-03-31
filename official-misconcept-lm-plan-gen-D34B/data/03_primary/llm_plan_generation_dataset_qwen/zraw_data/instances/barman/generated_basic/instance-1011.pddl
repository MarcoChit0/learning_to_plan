(define (problem prob)
 (:domain barman)
 (:objects 
      shaker334 - shaker
      left right - hand
      shot34 - shot
      ingredient255 ingredient455 ingredient168 - ingredient
      cocktail1 - cocktail
      dispenser56 dispenser359 dispenser15 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker334)
  (ontable shot34)
  (dispenses dispenser56 ingredient255)
  (dispenses dispenser359 ingredient455)
  (dispenses dispenser15 ingredient168)
  (clean shaker334)
  (clean shot34)
  (empty shaker334)
  (empty shot34)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker334 l0)
  (shaker-level shaker334 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient255)
  (cocktail-part2 cocktail1 ingredient168)
)
 (:goal
  (and
      (contains shot34 cocktail1)
)))
