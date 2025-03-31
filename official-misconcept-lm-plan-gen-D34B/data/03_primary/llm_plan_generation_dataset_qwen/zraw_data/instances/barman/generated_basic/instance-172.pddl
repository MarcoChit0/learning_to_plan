(define (problem prob)
 (:domain barman)
 (:objects 
      shaker55 - shaker
      left right - hand
      shot193 shot67 - shot
      ingredient484 ingredient217 ingredient435 - ingredient
      cocktail1 - cocktail
      dispenser372 dispenser283 dispenser41 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker55)
  (ontable shot193)
  (ontable shot67)
  (dispenses dispenser372 ingredient484)
  (dispenses dispenser283 ingredient217)
  (dispenses dispenser41 ingredient435)
  (clean shaker55)
  (clean shot193)
  (clean shot67)
  (empty shaker55)
  (empty shot193)
  (empty shot67)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker55 l0)
  (shaker-level shaker55 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient217)
  (cocktail-part2 cocktail1 ingredient435)
)
 (:goal
  (and
      (contains shot193 cocktail1)
)))
