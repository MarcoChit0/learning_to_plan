(define (problem prob)
 (:domain barman)
 (:objects 
      shaker316 - shaker
      left right - hand
      shot353 - shot
      ingredient292 ingredient247 ingredient211 - ingredient
      cocktail347 - cocktail
      dispenser91 dispenser78 dispenser189 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker316)
  (ontable shot353)
  (dispenses dispenser91 ingredient292)
  (dispenses dispenser78 ingredient247)
  (dispenses dispenser189 ingredient211)
  (clean shaker316)
  (clean shot353)
  (empty shaker316)
  (empty shot353)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker316 l0)
  (shaker-level shaker316 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail347 ingredient247)
  (cocktail-part2 cocktail347 ingredient211)
)
 (:goal
  (and
      (contains shot353 cocktail347)
)))
