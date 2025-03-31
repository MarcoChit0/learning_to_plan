(define (problem prob)
 (:domain barman)
 (:objects 
      shaker396 - shaker
      left right - hand
      shot329 shot448 - shot
      ingredient372 ingredient483 ingredient27 ingredient48 - ingredient
      cocktail83 - cocktail
      dispenser436 dispenser256 dispenser240 dispenser152 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker396)
  (ontable shot329)
  (ontable shot448)
  (dispenses dispenser436 ingredient372)
  (dispenses dispenser256 ingredient483)
  (dispenses dispenser240 ingredient27)
  (dispenses dispenser152 ingredient48)
  (clean shaker396)
  (clean shot329)
  (clean shot448)
  (empty shaker396)
  (empty shot329)
  (empty shot448)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker396 l0)
  (shaker-level shaker396 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail83 ingredient372)
  (cocktail-part2 cocktail83 ingredient483)
)
 (:goal
  (and
      (contains shot329 cocktail83)
)))
