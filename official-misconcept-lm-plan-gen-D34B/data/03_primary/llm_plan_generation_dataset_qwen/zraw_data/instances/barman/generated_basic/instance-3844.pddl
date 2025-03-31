(define (problem prob)
 (:domain barman)
 (:objects 
      shaker87 - shaker
      left right - hand
      shot61 shot395 shot182 - shot
      ingredient118 ingredient263 - ingredient
      cocktail463 - cocktail
      dispenser20 dispenser231 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker87)
  (ontable shot61)
  (ontable shot395)
  (ontable shot182)
  (dispenses dispenser20 ingredient118)
  (dispenses dispenser231 ingredient263)
  (clean shaker87)
  (clean shot61)
  (clean shot395)
  (clean shot182)
  (empty shaker87)
  (empty shot61)
  (empty shot395)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker87 l0)
  (shaker-level shaker87 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail463 ingredient263)
  (cocktail-part2 cocktail463 ingredient118)
)
 (:goal
  (and
      (contains shot61 cocktail463)
      (contains shot395 cocktail463)
)))
