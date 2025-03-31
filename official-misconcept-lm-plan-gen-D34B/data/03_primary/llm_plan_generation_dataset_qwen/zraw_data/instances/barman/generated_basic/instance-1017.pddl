(define (problem prob)
 (:domain barman)
 (:objects 
      shaker226 - shaker
      left right - hand
      shot329 - shot
      ingredient303 ingredient455 ingredient482 - ingredient
      cocktail1 - cocktail
      dispenser329 dispenser499 dispenser410 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker226)
  (ontable shot329)
  (dispenses dispenser329 ingredient303)
  (dispenses dispenser499 ingredient455)
  (dispenses dispenser410 ingredient482)
  (clean shaker226)
  (clean shot329)
  (empty shaker226)
  (empty shot329)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker226 l0)
  (shaker-level shaker226 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient455)
  (cocktail-part2 cocktail1 ingredient303)
)
 (:goal
  (and
      (contains shot329 cocktail1)
)))
