(define (problem prob)
 (:domain barman)
 (:objects 
      shaker227 - shaker
      left right - hand
      shot492 shot122 - shot
      ingredient361 ingredient42 - ingredient
      cocktail448 - cocktail
      dispenser290 dispenser248 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker227)
  (ontable shot492)
  (ontable shot122)
  (dispenses dispenser290 ingredient361)
  (dispenses dispenser248 ingredient42)
  (clean shaker227)
  (clean shot492)
  (clean shot122)
  (empty shaker227)
  (empty shot492)
  (empty shot122)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker227 l0)
  (shaker-level shaker227 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail448 ingredient42)
  (cocktail-part2 cocktail448 ingredient361)
)
 (:goal
  (and
      (contains shot492 cocktail448)
)))
