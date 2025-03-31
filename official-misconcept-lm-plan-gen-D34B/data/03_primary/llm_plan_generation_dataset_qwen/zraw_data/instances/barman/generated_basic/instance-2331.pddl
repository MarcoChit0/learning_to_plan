(define (problem prob)
 (:domain barman)
 (:objects 
      shaker277 - shaker
      left right - hand
      shot323 - shot
      ingredient389 ingredient101 - ingredient
      cocktail22 - cocktail
      dispenser251 dispenser331 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker277)
  (ontable shot323)
  (dispenses dispenser251 ingredient389)
  (dispenses dispenser331 ingredient101)
  (clean shaker277)
  (clean shot323)
  (empty shaker277)
  (empty shot323)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker277 l0)
  (shaker-level shaker277 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail22 ingredient389)
  (cocktail-part2 cocktail22 ingredient101)
)
 (:goal
  (and
      (contains shot323 cocktail22)
)))
