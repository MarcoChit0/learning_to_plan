(define (problem prob)
 (:domain barman)
 (:objects 
      shaker97 - shaker
      left right - hand
      shot155 - shot
      ingredient38 ingredient117 - ingredient
      cocktail1 - cocktail
      dispenser296 dispenser63 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker97)
  (ontable shot155)
  (dispenses dispenser296 ingredient38)
  (dispenses dispenser63 ingredient117)
  (clean shaker97)
  (clean shot155)
  (empty shaker97)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker97 l0)
  (shaker-level shaker97 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient117)
  (cocktail-part2 cocktail1 ingredient38)
)
 (:goal
  (and
      (contains shot155 cocktail1)
)))
