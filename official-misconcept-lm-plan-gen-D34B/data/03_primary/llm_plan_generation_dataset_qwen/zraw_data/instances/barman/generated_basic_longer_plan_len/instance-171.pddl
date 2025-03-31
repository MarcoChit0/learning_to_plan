(define (problem prob)
 (:domain barman)
 (:objects 
      shaker83 - shaker
      left right - hand
      shot424 shot110 shot61 shot272 - shot
      ingredient21 ingredient71 ingredient251 - ingredient
      cocktail1 - cocktail
      dispenser89 dispenser183 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker83)
  (ontable shot424)
  (ontable shot110)
  (ontable shot61)
  (ontable shot272)
  (dispenses dispenser89 ingredient21)
  (dispenses dispenser183 ingredient71)
  (dispenses dispenser492 ingredient251)
  (clean shaker83)
  (clean shot424)
  (clean shot110)
  (clean shot61)
  (clean shot272)
  (empty shaker83)
  (empty shot424)
  (empty shot110)
  (empty shot61)
  (empty shot272)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker83 l0)
  (shaker-level shaker83 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient71)
  (cocktail-part2 cocktail1 ingredient21)
)
 (:goal
  (and
      (contains shot424 cocktail1)
      (contains shot110 ingredient71)
      (contains shot61 ingredient71)
)))
