(define (problem prob)
 (:domain barman)
 (:objects 
      shaker8 - shaker
      left right - hand
      shot7 shot193 shot411 - shot
      ingredient281 ingredient211 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser298 dispenser344 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker8)
  (ontable shot7)
  (ontable shot193)
  (ontable shot411)
  (dispenses dispenser298 ingredient281)
  (dispenses dispenser344 ingredient211)
  (clean shaker8)
  (clean shot7)
  (clean shot193)
  (clean shot411)
  (empty shaker8)
  (empty shot7)
  (empty shot193)
  (empty shot411)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker8 l0)
  (shaker-level shaker8 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient281)
  (cocktail-part2 cocktail1 ingredient211)
  (cocktail-part1 cocktail2 ingredient281)
  (cocktail-part2 cocktail2 ingredient211)
)
 (:goal
  (and
      (contains shot7 cocktail2)
      (contains shot193 cocktail1)
)))
