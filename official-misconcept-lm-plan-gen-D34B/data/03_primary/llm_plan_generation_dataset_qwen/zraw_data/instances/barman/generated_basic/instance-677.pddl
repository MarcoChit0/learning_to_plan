(define (problem prob)
 (:domain barman)
 (:objects 
      shaker165 - shaker
      left right - hand
      shot288 shot130 - shot
      ingredient408 ingredient23 - ingredient
      cocktail1 - cocktail
      dispenser28 dispenser480 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker165)
  (ontable shot288)
  (ontable shot130)
  (dispenses dispenser28 ingredient408)
  (dispenses dispenser480 ingredient23)
  (clean shaker165)
  (clean shot288)
  (clean shot130)
  (empty shaker165)
  (empty shot288)
  (empty shot130)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker165 l0)
  (shaker-level shaker165 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient408)
  (cocktail-part2 cocktail1 ingredient23)
)
 (:goal
  (and
      (contains shot288 cocktail1)
)))
