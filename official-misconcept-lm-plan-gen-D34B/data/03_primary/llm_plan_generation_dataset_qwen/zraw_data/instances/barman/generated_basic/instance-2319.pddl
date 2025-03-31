(define (problem prob)
 (:domain barman)
 (:objects 
      shaker453 - shaker
      left right - hand
      shot142 - shot
      ingredient109 ingredient192 - ingredient
      cocktail299 - cocktail
      dispenser335 dispenser383 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker453)
  (ontable shot142)
  (dispenses dispenser335 ingredient109)
  (dispenses dispenser383 ingredient192)
  (clean shaker453)
  (clean shot142)
  (empty shaker453)
  (empty shot142)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker453 l0)
  (shaker-level shaker453 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail299 ingredient192)
  (cocktail-part2 cocktail299 ingredient109)
)
 (:goal
  (and
      (contains shot142 cocktail299)
)))
