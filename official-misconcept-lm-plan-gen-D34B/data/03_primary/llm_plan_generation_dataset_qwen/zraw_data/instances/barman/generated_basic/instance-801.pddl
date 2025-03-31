(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot25 - shot
      ingredient323 ingredient303 - ingredient
      cocktail1 - cocktail
      dispenser218 dispenser71 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot25)
  (dispenses dispenser218 ingredient323)
  (dispenses dispenser71 ingredient303)
  (clean shaker432)
  (clean shot25)
  (empty shaker432)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient303)
  (cocktail-part2 cocktail1 ingredient323)
)
 (:goal
  (and
      (contains shot25 cocktail1)
)))
