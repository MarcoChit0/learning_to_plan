(define (problem prob)
 (:domain barman)
 (:objects 
      shaker192 - shaker
      left right - hand
      shot155 shot5 - shot
      ingredient286 ingredient234 ingredient477 - ingredient
      cocktail1 - cocktail
      dispenser285 dispenser287 dispenser174 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker192)
  (ontable shot155)
  (ontable shot5)
  (dispenses dispenser285 ingredient286)
  (dispenses dispenser287 ingredient234)
  (dispenses dispenser174 ingredient477)
  (clean shaker192)
  (clean shot155)
  (clean shot5)
  (empty shaker192)
  (empty shot155)
  (empty shot5)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker192 l0)
  (shaker-level shaker192 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient234)
  (cocktail-part2 cocktail1 ingredient477)
)
 (:goal
  (and
      (contains shot155 cocktail1)
)))
