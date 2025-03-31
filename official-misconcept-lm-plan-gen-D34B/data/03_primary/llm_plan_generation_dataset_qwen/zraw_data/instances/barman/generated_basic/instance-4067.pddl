(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot417 shot416 - shot
      ingredient494 ingredient486 - ingredient
      cocktail5 - cocktail
      dispenser275 dispenser321 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot417)
  (ontable shot416)
  (dispenses dispenser275 ingredient494)
  (dispenses dispenser321 ingredient486)
  (clean shaker303)
  (clean shot417)
  (clean shot416)
  (empty shaker303)
  (empty shot417)
  (empty shot416)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail5 ingredient486)
  (cocktail-part2 cocktail5 ingredient494)
)
 (:goal
  (and
      (contains shot417 cocktail5)
)))
