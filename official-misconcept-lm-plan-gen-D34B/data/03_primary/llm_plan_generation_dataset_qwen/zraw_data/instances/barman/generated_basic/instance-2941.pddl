(define (problem prob)
 (:domain barman)
 (:objects 
      shaker188 - shaker
      left right - hand
      shot163 - shot
      ingredient478 ingredient499 ingredient385 ingredient286 - ingredient
      cocktail480 - cocktail
      dispenser289 dispenser258 dispenser182 dispenser83 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker188)
  (ontable shot163)
  (dispenses dispenser289 ingredient478)
  (dispenses dispenser258 ingredient499)
  (dispenses dispenser182 ingredient385)
  (dispenses dispenser83 ingredient286)
  (clean shaker188)
  (clean shot163)
  (empty shaker188)
  (empty shot163)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker188 l0)
  (shaker-level shaker188 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail480 ingredient478)
  (cocktail-part2 cocktail480 ingredient286)
)
 (:goal
  (and
      (contains shot163 cocktail480)
)))
