(define (problem prob)
 (:domain barman)
 (:objects 
      shaker197 - shaker
      left right - hand
      shot119 shot49 - shot
      ingredient451 ingredient142 - ingredient
      cocktail189 - cocktail
      dispenser118 dispenser313 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker197)
  (ontable shot119)
  (ontable shot49)
  (dispenses dispenser118 ingredient451)
  (dispenses dispenser313 ingredient142)
  (clean shaker197)
  (clean shot119)
  (clean shot49)
  (empty shaker197)
  (empty shot119)
  (empty shot49)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker197 l0)
  (shaker-level shaker197 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail189 ingredient142)
  (cocktail-part2 cocktail189 ingredient451)
)
 (:goal
  (and
      (contains shot119 cocktail189)
)))
