(define (problem prob)
 (:domain barman)
 (:objects 
      shaker342 - shaker
      left right - hand
      shot181 shot106 - shot
      ingredient119 ingredient213 ingredient304 - ingredient
      cocktail115 - cocktail
      dispenser72 dispenser104 dispenser163 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker342)
  (ontable shot181)
  (ontable shot106)
  (dispenses dispenser72 ingredient119)
  (dispenses dispenser104 ingredient213)
  (dispenses dispenser163 ingredient304)
  (clean shaker342)
  (clean shot181)
  (clean shot106)
  (empty shaker342)
  (empty shot181)
  (empty shot106)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker342 l0)
  (shaker-level shaker342 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail115 ingredient213)
  (cocktail-part2 cocktail115 ingredient304)
)
 (:goal
  (and
      (contains shot181 cocktail115)
)))
