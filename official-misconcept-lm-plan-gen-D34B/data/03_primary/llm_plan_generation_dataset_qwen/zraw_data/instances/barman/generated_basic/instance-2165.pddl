(define (problem prob)
 (:domain barman)
 (:objects 
      shaker298 - shaker
      left right - hand
      shot132 shot439 - shot
      ingredient290 ingredient280 - ingredient
      cocktail164 - cocktail
      dispenser173 dispenser187 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker298)
  (ontable shot132)
  (ontable shot439)
  (dispenses dispenser173 ingredient290)
  (dispenses dispenser187 ingredient280)
  (clean shaker298)
  (clean shot132)
  (clean shot439)
  (empty shaker298)
  (empty shot132)
  (empty shot439)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker298 l0)
  (shaker-level shaker298 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient280)
  (cocktail-part2 cocktail164 ingredient290)
)
 (:goal
  (and
      (contains shot132 cocktail164)
)))
