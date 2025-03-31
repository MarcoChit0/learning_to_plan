(define (problem prob)
 (:domain barman)
 (:objects 
      shaker47 - shaker
      left right - hand
      shot90 shot372 - shot
      ingredient92 ingredient224 ingredient16 - ingredient
      cocktail1 - cocktail
      dispenser330 dispenser209 dispenser355 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker47)
  (ontable shot90)
  (ontable shot372)
  (dispenses dispenser330 ingredient92)
  (dispenses dispenser209 ingredient224)
  (dispenses dispenser355 ingredient16)
  (clean shaker47)
  (clean shot90)
  (clean shot372)
  (empty shaker47)
  (empty shot90)
  (empty shot372)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker47 l0)
  (shaker-level shaker47 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient224)
  (cocktail-part2 cocktail1 ingredient92)
)
 (:goal
  (and
      (contains shot90 cocktail1)
)))
