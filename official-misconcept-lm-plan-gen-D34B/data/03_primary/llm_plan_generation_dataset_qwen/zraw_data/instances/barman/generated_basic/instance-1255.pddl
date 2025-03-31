(define (problem prob)
 (:domain barman)
 (:objects 
      shaker61 - shaker
      left right - hand
      shot490 shot95 shot401 - shot
      ingredient479 ingredient56 ingredient161 ingredient220 - ingredient
      cocktail465 - cocktail
      dispenser247 dispenser328 dispenser424 dispenser398 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker61)
  (ontable shot490)
  (ontable shot95)
  (ontable shot401)
  (dispenses dispenser247 ingredient479)
  (dispenses dispenser328 ingredient56)
  (dispenses dispenser424 ingredient161)
  (dispenses dispenser398 ingredient220)
  (clean shaker61)
  (clean shot490)
  (clean shot95)
  (clean shot401)
  (empty shaker61)
  (empty shot490)
  (empty shot95)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker61 l0)
  (shaker-level shaker61 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail465 ingredient479)
  (cocktail-part2 cocktail465 ingredient161)
)
 (:goal
  (and
      (contains shot490 cocktail465)
      (contains shot95 ingredient56)
)))
