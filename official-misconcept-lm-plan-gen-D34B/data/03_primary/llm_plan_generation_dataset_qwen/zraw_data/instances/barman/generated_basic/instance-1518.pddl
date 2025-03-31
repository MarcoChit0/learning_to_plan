(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot312 shot272 - shot
      ingredient253 ingredient36 ingredient410 ingredient120 - ingredient
      cocktail184 - cocktail
      dispenser23 dispenser234 dispenser188 dispenser270 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot312)
  (ontable shot272)
  (dispenses dispenser23 ingredient253)
  (dispenses dispenser234 ingredient36)
  (dispenses dispenser188 ingredient410)
  (dispenses dispenser270 ingredient120)
  (clean shaker266)
  (clean shot312)
  (clean shot272)
  (empty shaker266)
  (empty shot312)
  (empty shot272)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail184 ingredient36)
  (cocktail-part2 cocktail184 ingredient410)
)
 (:goal
  (and
      (contains shot312 cocktail184)
)))
