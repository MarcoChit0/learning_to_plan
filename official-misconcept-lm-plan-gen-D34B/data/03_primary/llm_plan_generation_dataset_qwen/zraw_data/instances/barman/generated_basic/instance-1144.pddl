(define (problem prob)
 (:domain barman)
 (:objects 
      shaker339 - shaker
      left right - hand
      shot450 shot213 - shot
      ingredient402 ingredient240 ingredient319 - ingredient
      cocktail1 - cocktail
      dispenser9 dispenser475 dispenser46 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker339)
  (ontable shot450)
  (ontable shot213)
  (dispenses dispenser9 ingredient402)
  (dispenses dispenser475 ingredient240)
  (dispenses dispenser46 ingredient319)
  (clean shaker339)
  (clean shot450)
  (clean shot213)
  (empty shaker339)
  (empty shot450)
  (empty shot213)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker339 l0)
  (shaker-level shaker339 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient319)
  (cocktail-part2 cocktail1 ingredient240)
)
 (:goal
  (and
      (contains shot450 cocktail1)
)))
