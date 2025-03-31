(define (problem prob)
 (:domain barman)
 (:objects 
      shaker383 - shaker
      left right - hand
      shot52 shot372 shot482 - shot
      ingredient193 ingredient340 ingredient269 - ingredient
      cocktail484 - cocktail
      dispenser392 dispenser356 dispenser53 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker383)
  (ontable shot52)
  (ontable shot372)
  (ontable shot482)
  (dispenses dispenser392 ingredient193)
  (dispenses dispenser356 ingredient340)
  (dispenses dispenser53 ingredient269)
  (clean shaker383)
  (clean shot52)
  (clean shot372)
  (clean shot482)
  (empty shaker383)
  (empty shot52)
  (empty shot372)
  (empty shot482)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker383 l0)
  (shaker-level shaker383 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail484 ingredient269)
  (cocktail-part2 cocktail484 ingredient340)
)
 (:goal
  (and
      (contains shot52 cocktail484)
      (contains shot372 ingredient269)
)))
