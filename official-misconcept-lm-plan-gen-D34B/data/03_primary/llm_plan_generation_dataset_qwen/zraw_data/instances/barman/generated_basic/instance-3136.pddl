(define (problem prob)
 (:domain barman)
 (:objects 
      shaker57 - shaker
      left right - hand
      shot412 shot207 shot8 - shot
      ingredient312 ingredient297 - ingredient
      cocktail100 - cocktail
      dispenser491 dispenser456 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker57)
  (ontable shot412)
  (ontable shot207)
  (ontable shot8)
  (dispenses dispenser491 ingredient312)
  (dispenses dispenser456 ingredient297)
  (clean shaker57)
  (clean shot412)
  (clean shot207)
  (clean shot8)
  (empty shaker57)
  (empty shot412)
  (empty shot207)
  (empty shot8)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker57 l0)
  (shaker-level shaker57 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail100 ingredient312)
  (cocktail-part2 cocktail100 ingredient297)
)
 (:goal
  (and
      (contains shot412 cocktail100)
      (contains shot207 ingredient297)
)))
