(define (problem prob)
 (:domain barman)
 (:objects 
      shaker362 - shaker
      left right - hand
      shot228 shot214 shot123 - shot
      ingredient6 ingredient473 ingredient394 - ingredient
      cocktail143 - cocktail
      dispenser70 dispenser69 dispenser44 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker362)
  (ontable shot228)
  (ontable shot214)
  (ontable shot123)
  (dispenses dispenser70 ingredient6)
  (dispenses dispenser69 ingredient473)
  (dispenses dispenser44 ingredient394)
  (clean shaker362)
  (clean shot228)
  (clean shot214)
  (clean shot123)
  (empty shaker362)
  (empty shot228)
  (empty shot214)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker362 l0)
  (shaker-level shaker362 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail143 ingredient6)
  (cocktail-part2 cocktail143 ingredient394)
)
 (:goal
  (and
      (contains shot228 cocktail143)
      (contains shot214 ingredient394)
)))
