(define (problem prob)
 (:domain barman)
 (:objects 
      shaker416 - shaker
      left right - hand
      shot255 - shot
      ingredient490 ingredient243 ingredient419 ingredient249 - ingredient
      cocktail1 - cocktail
      dispenser412 dispenser71 dispenser216 dispenser54 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker416)
  (ontable shot255)
  (dispenses dispenser412 ingredient490)
  (dispenses dispenser71 ingredient243)
  (dispenses dispenser216 ingredient419)
  (dispenses dispenser54 ingredient249)
  (clean shaker416)
  (clean shot255)
  (empty shaker416)
  (empty shot255)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker416 l0)
  (shaker-level shaker416 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient490)
  (cocktail-part2 cocktail1 ingredient419)
)
 (:goal
  (and
      (contains shot255 cocktail1)
)))
