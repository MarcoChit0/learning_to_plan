(define (problem prob)
 (:domain barman)
 (:objects 
      shaker368 - shaker
      left right - hand
      shot68 - shot
      ingredient203 ingredient477 ingredient186 - ingredient
      cocktail1 - cocktail
      dispenser93 dispenser370 dispenser430 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker368)
  (ontable shot68)
  (dispenses dispenser93 ingredient203)
  (dispenses dispenser370 ingredient477)
  (dispenses dispenser430 ingredient186)
  (clean shaker368)
  (clean shot68)
  (empty shaker368)
  (empty shot68)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker368 l0)
  (shaker-level shaker368 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient186)
  (cocktail-part2 cocktail1 ingredient203)
)
 (:goal
  (and
      (contains shot68 cocktail1)
)))
