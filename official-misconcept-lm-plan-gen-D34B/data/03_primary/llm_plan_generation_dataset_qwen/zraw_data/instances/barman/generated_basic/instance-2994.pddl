(define (problem prob)
 (:domain barman)
 (:objects 
      shaker374 - shaker
      left right - hand
      shot139 shot471 - shot
      ingredient91 ingredient299 ingredient255 ingredient152 - ingredient
      cocktail363 - cocktail
      dispenser212 dispenser191 dispenser437 dispenser243 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker374)
  (ontable shot139)
  (ontable shot471)
  (dispenses dispenser212 ingredient91)
  (dispenses dispenser191 ingredient299)
  (dispenses dispenser437 ingredient255)
  (dispenses dispenser243 ingredient152)
  (clean shaker374)
  (clean shot139)
  (clean shot471)
  (empty shaker374)
  (empty shot139)
  (empty shot471)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker374 l0)
  (shaker-level shaker374 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail363 ingredient91)
  (cocktail-part2 cocktail363 ingredient255)
)
 (:goal
  (and
      (contains shot139 cocktail363)
)))
