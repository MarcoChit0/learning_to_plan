(define (problem prob)
 (:domain barman)
 (:objects 
      shaker124 - shaker
      left right - hand
      shot380 shot460 - shot
      ingredient223 ingredient416 ingredient472 - ingredient
      cocktail42 - cocktail
      dispenser477 dispenser330 dispenser307 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker124)
  (ontable shot380)
  (ontable shot460)
  (dispenses dispenser477 ingredient223)
  (dispenses dispenser330 ingredient416)
  (dispenses dispenser307 ingredient472)
  (clean shaker124)
  (clean shot380)
  (clean shot460)
  (empty shaker124)
  (empty shot380)
  (empty shot460)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker124 l0)
  (shaker-level shaker124 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail42 ingredient416)
  (cocktail-part2 cocktail42 ingredient223)
)
 (:goal
  (and
      (contains shot380 cocktail42)
)))
