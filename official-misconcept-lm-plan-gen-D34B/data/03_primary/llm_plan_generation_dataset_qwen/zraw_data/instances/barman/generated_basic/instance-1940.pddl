(define (problem prob)
 (:domain barman)
 (:objects 
      shaker260 - shaker
      left right - hand
      shot90 - shot
      ingredient326 ingredient360 - ingredient
      cocktail482 - cocktail
      dispenser60 dispenser375 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker260)
  (ontable shot90)
  (dispenses dispenser60 ingredient326)
  (dispenses dispenser375 ingredient360)
  (clean shaker260)
  (clean shot90)
  (empty shaker260)
  (empty shot90)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker260 l0)
  (shaker-level shaker260 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient360)
  (cocktail-part2 cocktail482 ingredient326)
)
 (:goal
  (and
      (contains shot90 cocktail482)
)))
