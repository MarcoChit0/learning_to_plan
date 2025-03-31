(define (problem prob)
 (:domain barman)
 (:objects 
      shaker10 - shaker
      left right - hand
      shot242 shot475 - shot
      ingredient41 ingredient100 - ingredient
      cocktail405 - cocktail
      dispenser69 dispenser436 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker10)
  (ontable shot242)
  (ontable shot475)
  (dispenses dispenser69 ingredient41)
  (dispenses dispenser436 ingredient100)
  (clean shaker10)
  (clean shot242)
  (clean shot475)
  (empty shaker10)
  (empty shot242)
  (empty shot475)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker10 l0)
  (shaker-level shaker10 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail405 ingredient41)
  (cocktail-part2 cocktail405 ingredient100)
)
 (:goal
  (and
      (contains shot242 cocktail405)
)))
