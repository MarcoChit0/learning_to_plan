(define (problem prob)
 (:domain barman)
 (:objects 
      shaker135 - shaker
      left right - hand
      shot400 shot213 - shot
      ingredient304 ingredient254 ingredient88 ingredient435 - ingredient
      cocktail202 - cocktail
      dispenser315 dispenser286 dispenser416 dispenser140 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker135)
  (ontable shot400)
  (ontable shot213)
  (dispenses dispenser315 ingredient304)
  (dispenses dispenser286 ingredient254)
  (dispenses dispenser416 ingredient88)
  (dispenses dispenser140 ingredient435)
  (clean shaker135)
  (clean shot400)
  (clean shot213)
  (empty shaker135)
  (empty shot400)
  (empty shot213)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker135 l0)
  (shaker-level shaker135 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail202 ingredient435)
  (cocktail-part2 cocktail202 ingredient304)
)
 (:goal
  (and
      (contains shot400 cocktail202)
)))
