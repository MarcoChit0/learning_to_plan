(define (problem prob)
 (:domain barman)
 (:objects 
      shaker186 - shaker
      left right - hand
      shot101 - shot
      ingredient170 ingredient241 - ingredient
      cocktail1 - cocktail
      dispenser355 dispenser124 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker186)
  (ontable shot101)
  (dispenses dispenser355 ingredient170)
  (dispenses dispenser124 ingredient241)
  (clean shaker186)
  (clean shot101)
  (empty shaker186)
  (empty shot101)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker186 l0)
  (shaker-level shaker186 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient241)
  (cocktail-part2 cocktail1 ingredient170)
)
 (:goal
  (and
      (contains shot101 cocktail1)
)))
