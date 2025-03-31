(define (problem prob)
 (:domain barman)
 (:objects 
      shaker150 - shaker
      left right - hand
      shot334 shot92 - shot
      ingredient49 ingredient277 ingredient185 - ingredient
      cocktail383 - cocktail
      dispenser325 dispenser118 dispenser159 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker150)
  (ontable shot334)
  (ontable shot92)
  (dispenses dispenser325 ingredient49)
  (dispenses dispenser118 ingredient277)
  (dispenses dispenser159 ingredient185)
  (clean shaker150)
  (clean shot334)
  (clean shot92)
  (empty shaker150)
  (empty shot334)
  (empty shot92)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker150 l0)
  (shaker-level shaker150 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail383 ingredient49)
  (cocktail-part2 cocktail383 ingredient277)
)
 (:goal
  (and
      (contains shot334 cocktail383)
)))
