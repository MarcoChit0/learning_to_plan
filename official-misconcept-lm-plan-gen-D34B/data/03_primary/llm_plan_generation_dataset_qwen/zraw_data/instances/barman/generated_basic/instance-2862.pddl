(define (problem prob)
 (:domain barman)
 (:objects 
      shaker18 - shaker
      left right - hand
      shot490 shot483 - shot
      ingredient296 ingredient212 ingredient373 ingredient367 - ingredient
      cocktail82 - cocktail
      dispenser183 dispenser111 dispenser117 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker18)
  (ontable shot490)
  (ontable shot483)
  (dispenses dispenser183 ingredient296)
  (dispenses dispenser111 ingredient212)
  (dispenses dispenser117 ingredient373)
  (dispenses dispenser216 ingredient367)
  (clean shaker18)
  (clean shot490)
  (clean shot483)
  (empty shaker18)
  (empty shot490)
  (empty shot483)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker18 l0)
  (shaker-level shaker18 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail82 ingredient296)
  (cocktail-part2 cocktail82 ingredient367)
)
 (:goal
  (and
      (contains shot490 cocktail82)
)))
