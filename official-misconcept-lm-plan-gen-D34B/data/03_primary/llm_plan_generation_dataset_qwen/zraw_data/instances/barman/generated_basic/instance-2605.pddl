(define (problem prob)
 (:domain barman)
 (:objects 
      shaker351 - shaker
      left right - hand
      shot424 shot440 - shot
      ingredient312 ingredient432 ingredient254 - ingredient
      cocktail87 - cocktail
      dispenser235 dispenser285 dispenser6 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker351)
  (ontable shot424)
  (ontable shot440)
  (dispenses dispenser235 ingredient312)
  (dispenses dispenser285 ingredient432)
  (dispenses dispenser6 ingredient254)
  (clean shaker351)
  (clean shot424)
  (clean shot440)
  (empty shaker351)
  (empty shot424)
  (empty shot440)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker351 l0)
  (shaker-level shaker351 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail87 ingredient432)
  (cocktail-part2 cocktail87 ingredient312)
)
 (:goal
  (and
      (contains shot424 cocktail87)
)))
