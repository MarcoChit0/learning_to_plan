(define (problem prob)
 (:domain barman)
 (:objects 
      shaker125 - shaker
      left right - hand
      shot72 - shot
      ingredient217 ingredient353 ingredient183 - ingredient
      cocktail419 - cocktail
      dispenser91 dispenser282 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker125)
  (ontable shot72)
  (dispenses dispenser91 ingredient217)
  (dispenses dispenser282 ingredient353)
  (dispenses dispenser391 ingredient183)
  (clean shaker125)
  (clean shot72)
  (empty shaker125)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker125 l0)
  (shaker-level shaker125 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail419 ingredient183)
  (cocktail-part2 cocktail419 ingredient353)
)
 (:goal
  (and
      (contains shot72 cocktail419)
)))
