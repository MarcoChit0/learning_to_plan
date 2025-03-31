(define (problem prob)
 (:domain barman)
 (:objects 
      shaker395 - shaker
      left right - hand
      shot283 shot332 - shot
      ingredient376 ingredient44 - ingredient
      cocktail186 - cocktail
      dispenser436 dispenser484 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker395)
  (ontable shot283)
  (ontable shot332)
  (dispenses dispenser436 ingredient376)
  (dispenses dispenser484 ingredient44)
  (clean shaker395)
  (clean shot283)
  (clean shot332)
  (empty shaker395)
  (empty shot283)
  (empty shot332)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker395 l0)
  (shaker-level shaker395 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail186 ingredient376)
  (cocktail-part2 cocktail186 ingredient44)
)
 (:goal
  (and
      (contains shot283 cocktail186)
)))
