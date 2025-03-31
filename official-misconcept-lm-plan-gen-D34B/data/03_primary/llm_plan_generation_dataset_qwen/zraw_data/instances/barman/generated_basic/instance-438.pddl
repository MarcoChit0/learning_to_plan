(define (problem prob)
 (:domain barman)
 (:objects 
      shaker460 - shaker
      left right - hand
      shot184 shot476 - shot
      ingredient407 ingredient467 ingredient462 - ingredient
      cocktail1 - cocktail
      dispenser301 dispenser88 dispenser354 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker460)
  (ontable shot184)
  (ontable shot476)
  (dispenses dispenser301 ingredient407)
  (dispenses dispenser88 ingredient467)
  (dispenses dispenser354 ingredient462)
  (clean shaker460)
  (clean shot184)
  (clean shot476)
  (empty shaker460)
  (empty shot184)
  (empty shot476)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker460 l0)
  (shaker-level shaker460 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient467)
  (cocktail-part2 cocktail1 ingredient407)
)
 (:goal
  (and
      (contains shot184 cocktail1)
)))
