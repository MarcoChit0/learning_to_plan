(define (problem prob)
 (:domain barman)
 (:objects 
      shaker321 - shaker
      left right - hand
      shot147 shot376 - shot
      ingredient283 ingredient398 ingredient298 ingredient63 - ingredient
      cocktail249 - cocktail
      dispenser403 dispenser330 dispenser115 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker321)
  (ontable shot147)
  (ontable shot376)
  (dispenses dispenser403 ingredient283)
  (dispenses dispenser330 ingredient398)
  (dispenses dispenser115 ingredient298)
  (dispenses dispenser92 ingredient63)
  (clean shaker321)
  (clean shot147)
  (clean shot376)
  (empty shaker321)
  (empty shot147)
  (empty shot376)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker321 l0)
  (shaker-level shaker321 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail249 ingredient283)
  (cocktail-part2 cocktail249 ingredient63)
)
 (:goal
  (and
      (contains shot147 cocktail249)
)))
