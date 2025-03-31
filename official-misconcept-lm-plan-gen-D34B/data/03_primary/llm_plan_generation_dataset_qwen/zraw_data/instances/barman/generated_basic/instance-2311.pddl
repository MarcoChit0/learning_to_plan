(define (problem prob)
 (:domain barman)
 (:objects 
      shaker171 - shaker
      left right - hand
      shot98 - shot
      ingredient137 ingredient141 - ingredient
      cocktail283 - cocktail
      dispenser74 dispenser358 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker171)
  (ontable shot98)
  (dispenses dispenser74 ingredient137)
  (dispenses dispenser358 ingredient141)
  (clean shaker171)
  (clean shot98)
  (empty shaker171)
  (empty shot98)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker171 l0)
  (shaker-level shaker171 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail283 ingredient141)
  (cocktail-part2 cocktail283 ingredient137)
)
 (:goal
  (and
      (contains shot98 cocktail283)
)))
