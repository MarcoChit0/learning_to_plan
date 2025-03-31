(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot363 shot140 shot208 - shot
      ingredient30 ingredient150 - ingredient
      cocktail45 - cocktail
      dispenser372 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot363)
  (ontable shot140)
  (ontable shot208)
  (dispenses dispenser372 ingredient30)
  (dispenses dispenser385 ingredient150)
  (clean shaker459)
  (clean shot363)
  (clean shot140)
  (clean shot208)
  (empty shaker459)
  (empty shot363)
  (empty shot140)
  (empty shot208)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail45 ingredient150)
  (cocktail-part2 cocktail45 ingredient30)
)
 (:goal
  (and
      (contains shot363 cocktail45)
      (contains shot140 ingredient30)
)))
