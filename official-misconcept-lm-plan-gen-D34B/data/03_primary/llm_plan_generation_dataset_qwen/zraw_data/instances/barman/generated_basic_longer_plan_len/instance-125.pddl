(define (problem prob)
 (:domain barman)
 (:objects 
      shaker271 - shaker
      left right - hand
      shot197 shot435 shot357 - shot
      ingredient471 ingredient309 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser274 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker271)
  (ontable shot197)
  (ontable shot435)
  (ontable shot357)
  (dispenses dispenser274 ingredient471)
  (dispenses dispenser283 ingredient309)
  (clean shaker271)
  (clean shot197)
  (clean shot435)
  (clean shot357)
  (empty shaker271)
  (empty shot197)
  (empty shot435)
  (empty shot357)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker271 l0)
  (shaker-level shaker271 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient471)
  (cocktail-part2 cocktail1 ingredient309)
  (cocktail-part1 cocktail2 ingredient309)
  (cocktail-part2 cocktail2 ingredient471)
)
 (:goal
  (and
      (contains shot197 cocktail1)
      (contains shot435 cocktail2)
)))
