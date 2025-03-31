(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot258 shot325 - shot
      ingredient298 ingredient75 - ingredient
      cocktail1 - cocktail
      dispenser14 dispenser442 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot258)
  (ontable shot325)
  (dispenses dispenser14 ingredient298)
  (dispenses dispenser442 ingredient75)
  (clean shaker332)
  (clean shot258)
  (clean shot325)
  (empty shaker332)
  (empty shot258)
  (empty shot325)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient298)
  (cocktail-part2 cocktail1 ingredient75)
)
 (:goal
  (and
      (contains shot258 cocktail1)
)))
