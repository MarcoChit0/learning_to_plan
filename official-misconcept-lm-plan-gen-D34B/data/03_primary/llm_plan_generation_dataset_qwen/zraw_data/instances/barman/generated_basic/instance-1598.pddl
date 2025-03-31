(define (problem prob)
 (:domain barman)
 (:objects 
      shaker497 - shaker
      left right - hand
      shot330 shot455 - shot
      ingredient368 ingredient41 - ingredient
      cocktail405 - cocktail
      dispenser438 dispenser15 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker497)
  (ontable shot330)
  (ontable shot455)
  (dispenses dispenser438 ingredient368)
  (dispenses dispenser15 ingredient41)
  (clean shaker497)
  (clean shot330)
  (clean shot455)
  (empty shaker497)
  (empty shot330)
  (empty shot455)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker497 l0)
  (shaker-level shaker497 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail405 ingredient41)
  (cocktail-part2 cocktail405 ingredient368)
)
 (:goal
  (and
      (contains shot330 cocktail405)
)))
