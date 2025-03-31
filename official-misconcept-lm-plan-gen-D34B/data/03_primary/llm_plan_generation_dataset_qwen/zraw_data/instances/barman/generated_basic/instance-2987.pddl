(define (problem prob)
 (:domain barman)
 (:objects 
      shaker291 - shaker
      left right - hand
      shot370 shot13 - shot
      ingredient435 ingredient48 ingredient9 ingredient395 - ingredient
      cocktail321 - cocktail
      dispenser140 dispenser11 dispenser378 dispenser219 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker291)
  (ontable shot370)
  (ontable shot13)
  (dispenses dispenser140 ingredient435)
  (dispenses dispenser11 ingredient48)
  (dispenses dispenser378 ingredient9)
  (dispenses dispenser219 ingredient395)
  (clean shaker291)
  (clean shot370)
  (clean shot13)
  (empty shaker291)
  (empty shot370)
  (empty shot13)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker291 l0)
  (shaker-level shaker291 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail321 ingredient9)
  (cocktail-part2 cocktail321 ingredient48)
)
 (:goal
  (and
      (contains shot370 cocktail321)
)))
