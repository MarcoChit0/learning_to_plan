(define (problem prob)
 (:domain barman)
 (:objects 
      shaker114 - shaker
      left right - hand
      shot200 shot110 shot192 - shot
      ingredient499 ingredient20 ingredient62 ingredient243 - ingredient
      cocktail1 - cocktail
      dispenser161 dispenser242 dispenser272 dispenser419 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker114)
  (ontable shot200)
  (ontable shot110)
  (ontable shot192)
  (dispenses dispenser161 ingredient499)
  (dispenses dispenser242 ingredient20)
  (dispenses dispenser272 ingredient62)
  (dispenses dispenser419 ingredient243)
  (clean shaker114)
  (clean shot200)
  (clean shot110)
  (clean shot192)
  (empty shaker114)
  (empty shot200)
  (empty shot110)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker114 l0)
  (shaker-level shaker114 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient499)
  (cocktail-part2 cocktail1 ingredient20)
)
 (:goal
  (and
      (contains shot200 cocktail1)
      (contains shot110 cocktail1)
)))
