(define (problem prob)
 (:domain barman)
 (:objects 
      shaker265 - shaker
      left right - hand
      shot219 - shot
      ingredient153 ingredient139 - ingredient
      cocktail252 - cocktail
      dispenser251 dispenser360 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker265)
  (ontable shot219)
  (dispenses dispenser251 ingredient153)
  (dispenses dispenser360 ingredient139)
  (clean shaker265)
  (clean shot219)
  (empty shaker265)
  (empty shot219)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker265 l0)
  (shaker-level shaker265 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail252 ingredient139)
  (cocktail-part2 cocktail252 ingredient153)
)
 (:goal
  (and
      (contains shot219 cocktail252)
)))
