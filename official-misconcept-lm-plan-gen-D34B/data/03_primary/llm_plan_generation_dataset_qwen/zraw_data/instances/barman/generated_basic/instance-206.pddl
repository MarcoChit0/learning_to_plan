(define (problem prob)
 (:domain barman)
 (:objects 
      shaker54 - shaker
      left right - hand
      shot315 - shot
      ingredient450 ingredient422 ingredient132 - ingredient
      cocktail1 - cocktail
      dispenser303 dispenser327 dispenser342 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker54)
  (ontable shot315)
  (dispenses dispenser303 ingredient450)
  (dispenses dispenser327 ingredient422)
  (dispenses dispenser342 ingredient132)
  (clean shaker54)
  (clean shot315)
  (empty shaker54)
  (empty shot315)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker54 l0)
  (shaker-level shaker54 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient450)
  (cocktail-part2 cocktail1 ingredient422)
)
 (:goal
  (and
      (contains shot315 cocktail1)
)))
