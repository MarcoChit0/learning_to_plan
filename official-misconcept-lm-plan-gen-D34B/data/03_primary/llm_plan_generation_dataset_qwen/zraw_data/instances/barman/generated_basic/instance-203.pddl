(define (problem prob)
 (:domain barman)
 (:objects 
      shaker374 - shaker
      left right - hand
      shot172 - shot
      ingredient165 ingredient271 ingredient335 - ingredient
      cocktail1 - cocktail
      dispenser286 dispenser304 dispenser322 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker374)
  (ontable shot172)
  (dispenses dispenser286 ingredient165)
  (dispenses dispenser304 ingredient271)
  (dispenses dispenser322 ingredient335)
  (clean shaker374)
  (clean shot172)
  (empty shaker374)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker374 l0)
  (shaker-level shaker374 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient271)
  (cocktail-part2 cocktail1 ingredient335)
)
 (:goal
  (and
      (contains shot172 cocktail1)
)))
