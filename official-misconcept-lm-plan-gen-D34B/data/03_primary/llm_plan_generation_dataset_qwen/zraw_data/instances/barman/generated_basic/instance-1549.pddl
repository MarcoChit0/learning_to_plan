(define (problem prob)
 (:domain barman)
 (:objects 
      shaker161 - shaker
      left right - hand
      shot302 - shot
      ingredient302 ingredient68 - ingredient
      cocktail339 - cocktail
      dispenser116 dispenser381 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker161)
  (ontable shot302)
  (dispenses dispenser116 ingredient302)
  (dispenses dispenser381 ingredient68)
  (clean shaker161)
  (clean shot302)
  (empty shaker161)
  (empty shot302)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker161 l0)
  (shaker-level shaker161 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail339 ingredient68)
  (cocktail-part2 cocktail339 ingredient302)
)
 (:goal
  (and
      (contains shot302 cocktail339)
)))
