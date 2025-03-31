(define (problem prob)
 (:domain barman)
 (:objects 
      shaker213 - shaker
      left right - hand
      shot13 - shot
      ingredient62 ingredient23 ingredient228 - ingredient
      cocktail103 - cocktail
      dispenser229 dispenser181 dispenser147 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker213)
  (ontable shot13)
  (dispenses dispenser229 ingredient62)
  (dispenses dispenser181 ingredient23)
  (dispenses dispenser147 ingredient228)
  (clean shaker213)
  (clean shot13)
  (empty shaker213)
  (empty shot13)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker213 l0)
  (shaker-level shaker213 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail103 ingredient62)
  (cocktail-part2 cocktail103 ingredient228)
)
 (:goal
  (and
      (contains shot13 cocktail103)
)))
