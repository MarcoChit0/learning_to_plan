(define (problem prob)
 (:domain barman)
 (:objects 
      shaker54 - shaker
      left right - hand
      shot44 shot76 - shot
      ingredient236 ingredient430 ingredient341 ingredient198 - ingredient
      cocktail1 - cocktail
      dispenser95 dispenser69 dispenser119 dispenser6 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker54)
  (ontable shot44)
  (ontable shot76)
  (dispenses dispenser95 ingredient236)
  (dispenses dispenser69 ingredient430)
  (dispenses dispenser119 ingredient341)
  (dispenses dispenser6 ingredient198)
  (clean shaker54)
  (clean shot44)
  (clean shot76)
  (empty shaker54)
  (empty shot44)
  (empty shot76)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker54 l0)
  (shaker-level shaker54 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient236)
  (cocktail-part2 cocktail1 ingredient198)
)
 (:goal
  (and
      (contains shot44 cocktail1)
)))
