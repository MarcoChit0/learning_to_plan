(define (problem prob)
 (:domain barman)
 (:objects 
      shaker334 - shaker
      left right - hand
      shot404 shot477 - shot
      ingredient281 ingredient182 - ingredient
      cocktail1 - cocktail
      dispenser411 dispenser329 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker334)
  (ontable shot404)
  (ontable shot477)
  (dispenses dispenser411 ingredient281)
  (dispenses dispenser329 ingredient182)
  (clean shaker334)
  (clean shot404)
  (clean shot477)
  (empty shaker334)
  (empty shot404)
  (empty shot477)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker334 l0)
  (shaker-level shaker334 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient281)
  (cocktail-part2 cocktail1 ingredient182)
)
 (:goal
  (and
      (contains shot404 cocktail1)
)))
