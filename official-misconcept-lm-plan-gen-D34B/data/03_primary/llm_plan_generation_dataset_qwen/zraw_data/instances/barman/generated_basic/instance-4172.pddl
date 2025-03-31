(define (problem prob)
 (:domain barman)
 (:objects 
      shaker317 - shaker
      left right - hand
      shot106 shot424 - shot
      ingredient49 ingredient434 - ingredient
      cocktail444 - cocktail
      dispenser398 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker317)
  (ontable shot106)
  (ontable shot424)
  (dispenses dispenser398 ingredient49)
  (dispenses dispenser149 ingredient434)
  (clean shaker317)
  (clean shot106)
  (clean shot424)
  (empty shaker317)
  (empty shot106)
  (empty shot424)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker317 l0)
  (shaker-level shaker317 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail444 ingredient434)
  (cocktail-part2 cocktail444 ingredient49)
)
 (:goal
  (and
      (contains shot106 cocktail444)
)))
