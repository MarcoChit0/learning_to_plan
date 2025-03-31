(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot436 shot131 - shot
      ingredient329 ingredient434 ingredient334 - ingredient
      cocktail480 - cocktail
      dispenser207 dispenser469 dispenser119 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot436)
  (ontable shot131)
  (dispenses dispenser207 ingredient329)
  (dispenses dispenser469 ingredient434)
  (dispenses dispenser119 ingredient334)
  (clean shaker228)
  (clean shot436)
  (clean shot131)
  (empty shaker228)
  (empty shot436)
  (empty shot131)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail480 ingredient434)
  (cocktail-part2 cocktail480 ingredient329)
)
 (:goal
  (and
      (contains shot436 cocktail480)
)))
