(define (problem prob)
 (:domain barman)
 (:objects 
      shaker89 - shaker
      left right - hand
      shot169 - shot
      ingredient377 ingredient287 ingredient407 ingredient118 - ingredient
      cocktail360 - cocktail
      dispenser143 dispenser163 dispenser217 dispenser430 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker89)
  (ontable shot169)
  (dispenses dispenser143 ingredient377)
  (dispenses dispenser163 ingredient287)
  (dispenses dispenser217 ingredient407)
  (dispenses dispenser430 ingredient118)
  (clean shaker89)
  (clean shot169)
  (empty shaker89)
  (empty shot169)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker89 l0)
  (shaker-level shaker89 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail360 ingredient287)
  (cocktail-part2 cocktail360 ingredient407)
)
 (:goal
  (and
      (contains shot169 cocktail360)
)))
