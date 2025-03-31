(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot39 - shot
      ingredient94 ingredient303 ingredient217 - ingredient
      cocktail1 - cocktail
      dispenser150 dispenser67 dispenser362 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot39)
  (dispenses dispenser150 ingredient94)
  (dispenses dispenser67 ingredient303)
  (dispenses dispenser362 ingredient217)
  (clean shaker209)
  (clean shot39)
  (empty shaker209)
  (empty shot39)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient94)
  (cocktail-part2 cocktail1 ingredient217)
)
 (:goal
  (and
      (contains shot39 cocktail1)
)))
