(define (problem prob)
 (:domain barman)
 (:objects 
      shaker290 - shaker
      left right - hand
      shot451 shot310 - shot
      ingredient423 ingredient92 ingredient196 - ingredient
      cocktail141 - cocktail
      dispenser381 dispenser412 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker290)
  (ontable shot451)
  (ontable shot310)
  (dispenses dispenser381 ingredient423)
  (dispenses dispenser412 ingredient92)
  (dispenses dispenser370 ingredient196)
  (clean shaker290)
  (clean shot451)
  (clean shot310)
  (empty shaker290)
  (empty shot451)
  (empty shot310)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker290 l0)
  (shaker-level shaker290 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail141 ingredient196)
  (cocktail-part2 cocktail141 ingredient92)
)
 (:goal
  (and
      (contains shot451 cocktail141)
)))
