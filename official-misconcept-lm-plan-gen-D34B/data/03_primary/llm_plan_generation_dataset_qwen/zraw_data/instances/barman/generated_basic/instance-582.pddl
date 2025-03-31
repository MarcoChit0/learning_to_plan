(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot376 shot130 - shot
      ingredient125 ingredient196 - ingredient
      cocktail1 - cocktail
      dispenser48 dispenser459 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot376)
  (ontable shot130)
  (dispenses dispenser48 ingredient125)
  (dispenses dispenser459 ingredient196)
  (clean shaker228)
  (clean shot376)
  (clean shot130)
  (empty shaker228)
  (empty shot376)
  (empty shot130)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient196)
  (cocktail-part2 cocktail1 ingredient125)
)
 (:goal
  (and
      (contains shot376 cocktail1)
)))
