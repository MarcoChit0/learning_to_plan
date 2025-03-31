(define (problem prob)
 (:domain barman)
 (:objects 
      shaker198 - shaker
      left right - hand
      shot434 - shot
      ingredient269 ingredient248 ingredient461 ingredient251 - ingredient
      cocktail21 - cocktail
      dispenser185 dispenser41 dispenser42 dispenser14 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker198)
  (ontable shot434)
  (dispenses dispenser185 ingredient269)
  (dispenses dispenser41 ingredient248)
  (dispenses dispenser42 ingredient461)
  (dispenses dispenser14 ingredient251)
  (clean shaker198)
  (clean shot434)
  (empty shaker198)
  (empty shot434)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker198 l0)
  (shaker-level shaker198 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail21 ingredient269)
  (cocktail-part2 cocktail21 ingredient461)
)
 (:goal
  (and
      (contains shot434 cocktail21)
)))
