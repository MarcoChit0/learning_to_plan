(define (problem prob)
 (:domain barman)
 (:objects 
      shaker414 - shaker
      left right - hand
      shot133 - shot
      ingredient275 ingredient43 - ingredient
      cocktail1 - cocktail
      dispenser151 dispenser287 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker414)
  (ontable shot133)
  (dispenses dispenser151 ingredient275)
  (dispenses dispenser287 ingredient43)
  (clean shaker414)
  (clean shot133)
  (empty shaker414)
  (empty shot133)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker414 l0)
  (shaker-level shaker414 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient43)
  (cocktail-part2 cocktail1 ingredient275)
)
 (:goal
  (and
      (contains shot133 cocktail1)
)))
