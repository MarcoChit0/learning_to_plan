(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot44 - shot
      ingredient17 ingredient41 - ingredient
      cocktail256 - cocktail
      dispenser183 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot44)
  (dispenses dispenser183 ingredient17)
  (dispenses dispenser130 ingredient41)
  (clean shaker305)
  (clean shot44)
  (empty shaker305)
  (empty shot44)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail256 ingredient17)
  (cocktail-part2 cocktail256 ingredient41)
)
 (:goal
  (and
      (contains shot44 cocktail256)
)))
