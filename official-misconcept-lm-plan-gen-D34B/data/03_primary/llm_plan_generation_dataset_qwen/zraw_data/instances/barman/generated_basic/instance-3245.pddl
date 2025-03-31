(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot439 - shot
      ingredient131 ingredient351 ingredient98 - ingredient
      cocktail310 - cocktail
      dispenser143 dispenser359 dispenser129 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot439)
  (dispenses dispenser143 ingredient131)
  (dispenses dispenser359 ingredient351)
  (dispenses dispenser129 ingredient98)
  (clean shaker426)
  (clean shot439)
  (empty shaker426)
  (empty shot439)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail310 ingredient98)
  (cocktail-part2 cocktail310 ingredient131)
)
 (:goal
  (and
      (contains shot439 cocktail310)
)))
