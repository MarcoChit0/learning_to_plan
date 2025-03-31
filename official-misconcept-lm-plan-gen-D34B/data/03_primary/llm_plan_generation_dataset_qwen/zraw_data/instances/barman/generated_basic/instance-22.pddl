(define (problem prob)
 (:domain barman)
 (:objects 
      shaker186 - shaker
      left right - hand
      shot356 shot408 - shot
      ingredient489 ingredient461 - ingredient
      cocktail1 - cocktail
      dispenser111 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker186)
  (ontable shot356)
  (ontable shot408)
  (dispenses dispenser111 ingredient489)
  (dispenses dispenser446 ingredient461)
  (clean shaker186)
  (clean shot356)
  (clean shot408)
  (empty shaker186)
  (empty shot356)
  (empty shot408)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker186 l0)
  (shaker-level shaker186 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient489)
  (cocktail-part2 cocktail1 ingredient461)
)
 (:goal
  (and
      (contains shot356 cocktail1)
)))
