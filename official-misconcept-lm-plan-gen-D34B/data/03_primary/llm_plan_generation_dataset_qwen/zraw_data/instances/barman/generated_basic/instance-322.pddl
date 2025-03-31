(define (problem prob)
 (:domain barman)
 (:objects 
      shaker399 - shaker
      left right - hand
      shot230 - shot
      ingredient55 ingredient9 - ingredient
      cocktail1 - cocktail
      dispenser462 dispenser119 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker399)
  (ontable shot230)
  (dispenses dispenser462 ingredient55)
  (dispenses dispenser119 ingredient9)
  (clean shaker399)
  (clean shot230)
  (empty shaker399)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker399 l0)
  (shaker-level shaker399 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient9)
  (cocktail-part2 cocktail1 ingredient55)
)
 (:goal
  (and
      (contains shot230 cocktail1)
)))
