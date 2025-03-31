(define (problem prob)
 (:domain barman)
 (:objects 
      shaker466 - shaker
      left right - hand
      shot386 shot363 shot143 - shot
      ingredient16 ingredient327 ingredient454 ingredient155 - ingredient
      cocktail439 - cocktail
      dispenser170 dispenser343 dispenser299 dispenser287 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker466)
  (ontable shot386)
  (ontable shot363)
  (ontable shot143)
  (dispenses dispenser170 ingredient16)
  (dispenses dispenser343 ingredient327)
  (dispenses dispenser299 ingredient454)
  (dispenses dispenser287 ingredient155)
  (clean shaker466)
  (clean shot386)
  (clean shot363)
  (clean shot143)
  (empty shaker466)
  (empty shot386)
  (empty shot363)
  (empty shot143)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker466 l0)
  (shaker-level shaker466 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail439 ingredient327)
  (cocktail-part2 cocktail439 ingredient16)
)
 (:goal
  (and
      (contains shot386 cocktail439)
      (contains shot363 cocktail439)
)))
