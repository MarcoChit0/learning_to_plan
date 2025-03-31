(define (problem prob)
 (:domain barman)
 (:objects 
      shaker73 - shaker
      left right - hand
      shot44 shot472 - shot
      ingredient204 ingredient428 - ingredient
      cocktail333 - cocktail
      dispenser280 dispenser340 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker73)
  (ontable shot44)
  (ontable shot472)
  (dispenses dispenser280 ingredient204)
  (dispenses dispenser340 ingredient428)
  (clean shaker73)
  (clean shot44)
  (clean shot472)
  (empty shaker73)
  (empty shot44)
  (empty shot472)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker73 l0)
  (shaker-level shaker73 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail333 ingredient428)
  (cocktail-part2 cocktail333 ingredient204)
)
 (:goal
  (and
      (contains shot44 cocktail333)
)))
