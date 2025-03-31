(define (problem prob)
 (:domain barman)
 (:objects 
      shaker52 - shaker
      left right - hand
      shot147 - shot
      ingredient248 ingredient141 - ingredient
      cocktail487 - cocktail
      dispenser261 dispenser396 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker52)
  (ontable shot147)
  (dispenses dispenser261 ingredient248)
  (dispenses dispenser396 ingredient141)
  (clean shaker52)
  (clean shot147)
  (empty shaker52)
  (empty shot147)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker52 l0)
  (shaker-level shaker52 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail487 ingredient248)
  (cocktail-part2 cocktail487 ingredient141)
)
 (:goal
  (and
      (contains shot147 cocktail487)
)))
