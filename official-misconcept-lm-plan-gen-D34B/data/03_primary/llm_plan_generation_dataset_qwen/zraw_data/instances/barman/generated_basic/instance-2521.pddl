(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot104 - shot
      ingredient304 ingredient357 - ingredient
      cocktail110 - cocktail
      dispenser64 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot104)
  (dispenses dispenser64 ingredient304)
  (dispenses dispenser431 ingredient357)
  (clean shaker318)
  (clean shot104)
  (empty shaker318)
  (empty shot104)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail110 ingredient304)
  (cocktail-part2 cocktail110 ingredient357)
)
 (:goal
  (and
      (contains shot104 cocktail110)
)))
