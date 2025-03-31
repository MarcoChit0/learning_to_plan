(define (problem prob)
 (:domain barman)
 (:objects 
      shaker325 - shaker
      left right - hand
      shot218 shot155 shot130 - shot
      ingredient99 ingredient297 ingredient399 - ingredient
      cocktail448 - cocktail
      dispenser65 dispenser87 dispenser272 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker325)
  (ontable shot218)
  (ontable shot155)
  (ontable shot130)
  (dispenses dispenser65 ingredient99)
  (dispenses dispenser87 ingredient297)
  (dispenses dispenser272 ingredient399)
  (clean shaker325)
  (clean shot218)
  (clean shot155)
  (clean shot130)
  (empty shaker325)
  (empty shot218)
  (empty shot155)
  (empty shot130)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker325 l0)
  (shaker-level shaker325 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail448 ingredient297)
  (cocktail-part2 cocktail448 ingredient99)
)
 (:goal
  (and
      (contains shot218 cocktail448)
      (contains shot155 ingredient297)
)))
