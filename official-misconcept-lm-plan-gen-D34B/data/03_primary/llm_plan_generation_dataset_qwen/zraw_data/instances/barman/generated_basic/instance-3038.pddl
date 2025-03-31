(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot244 - shot
      ingredient193 ingredient178 ingredient206 - ingredient
      cocktail441 - cocktail
      dispenser194 dispenser471 dispenser227 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot244)
  (dispenses dispenser194 ingredient193)
  (dispenses dispenser471 ingredient178)
  (dispenses dispenser227 ingredient206)
  (clean shaker64)
  (clean shot244)
  (empty shaker64)
  (empty shot244)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail441 ingredient193)
  (cocktail-part2 cocktail441 ingredient178)
)
 (:goal
  (and
      (contains shot244 cocktail441)
)))
