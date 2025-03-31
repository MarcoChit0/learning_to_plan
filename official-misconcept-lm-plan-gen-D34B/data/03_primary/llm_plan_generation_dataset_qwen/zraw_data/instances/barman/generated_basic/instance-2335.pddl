(define (problem prob)
 (:domain barman)
 (:objects 
      shaker4 - shaker
      left right - hand
      shot428 - shot
      ingredient406 ingredient107 - ingredient
      cocktail162 - cocktail
      dispenser227 dispenser437 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker4)
  (ontable shot428)
  (dispenses dispenser227 ingredient406)
  (dispenses dispenser437 ingredient107)
  (clean shaker4)
  (clean shot428)
  (empty shaker4)
  (empty shot428)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker4 l0)
  (shaker-level shaker4 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail162 ingredient107)
  (cocktail-part2 cocktail162 ingredient406)
)
 (:goal
  (and
      (contains shot428 cocktail162)
)))
