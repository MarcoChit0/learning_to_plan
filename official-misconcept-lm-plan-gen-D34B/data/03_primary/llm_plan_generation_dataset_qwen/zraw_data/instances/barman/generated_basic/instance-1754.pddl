(define (problem prob)
 (:domain barman)
 (:objects 
      shaker493 - shaker
      left right - hand
      shot483 shot426 shot140 - shot
      ingredient377 ingredient428 - ingredient
      cocktail172 - cocktail
      dispenser325 dispenser3 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker493)
  (ontable shot483)
  (ontable shot426)
  (ontable shot140)
  (dispenses dispenser325 ingredient377)
  (dispenses dispenser3 ingredient428)
  (clean shaker493)
  (clean shot483)
  (clean shot426)
  (clean shot140)
  (empty shaker493)
  (empty shot483)
  (empty shot426)
  (empty shot140)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker493 l0)
  (shaker-level shaker493 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail172 ingredient377)
  (cocktail-part2 cocktail172 ingredient428)
)
 (:goal
  (and
      (contains shot483 cocktail172)
      (contains shot426 ingredient428)
)))
