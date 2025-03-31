(define (problem prob)
 (:domain barman)
 (:objects 
      shaker174 - shaker
      left right - hand
      shot373 shot194 - shot
      ingredient225 ingredient257 ingredient268 - ingredient
      cocktail125 - cocktail
      dispenser307 dispenser263 dispenser462 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker174)
  (ontable shot373)
  (ontable shot194)
  (dispenses dispenser307 ingredient225)
  (dispenses dispenser263 ingredient257)
  (dispenses dispenser462 ingredient268)
  (clean shaker174)
  (clean shot373)
  (clean shot194)
  (empty shaker174)
  (empty shot373)
  (empty shot194)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker174 l0)
  (shaker-level shaker174 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail125 ingredient225)
  (cocktail-part2 cocktail125 ingredient268)
)
 (:goal
  (and
      (contains shot373 cocktail125)
)))
