(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot145 - shot
      ingredient103 ingredient433 - ingredient
      cocktail1 - cocktail
      dispenser65 dispenser412 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot145)
  (dispenses dispenser65 ingredient103)
  (dispenses dispenser412 ingredient433)
  (clean shaker273)
  (clean shot145)
  (empty shaker273)
  (empty shot145)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient433)
  (cocktail-part2 cocktail1 ingredient103)
)
 (:goal
  (and
      (contains shot145 cocktail1)
)))
