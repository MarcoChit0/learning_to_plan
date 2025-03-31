(define (problem prob)
 (:domain barman)
 (:objects 
      shaker171 - shaker
      left right - hand
      shot124 shot87 - shot
      ingredient397 ingredient159 ingredient209 ingredient103 - ingredient
      cocktail1 - cocktail
      dispenser370 dispenser313 dispenser287 dispenser356 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker171)
  (ontable shot124)
  (ontable shot87)
  (dispenses dispenser370 ingredient397)
  (dispenses dispenser313 ingredient159)
  (dispenses dispenser287 ingredient209)
  (dispenses dispenser356 ingredient103)
  (clean shaker171)
  (clean shot124)
  (clean shot87)
  (empty shaker171)
  (empty shot124)
  (empty shot87)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker171 l0)
  (shaker-level shaker171 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient397)
  (cocktail-part2 cocktail1 ingredient209)
)
 (:goal
  (and
      (contains shot124 cocktail1)
)))
