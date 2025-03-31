(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot395 shot213 - shot
      ingredient266 ingredient466 - ingredient
      cocktail1 - cocktail
      dispenser197 dispenser332 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot395)
  (ontable shot213)
  (dispenses dispenser197 ingredient266)
  (dispenses dispenser332 ingredient466)
  (clean shaker294)
  (clean shot395)
  (clean shot213)
  (empty shaker294)
  (empty shot395)
  (empty shot213)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient466)
  (cocktail-part2 cocktail1 ingredient266)
)
 (:goal
  (and
      (contains shot395 cocktail1)
)))
