(define (problem prob)
 (:domain barman)
 (:objects 
      shaker235 - shaker
      left right - hand
      shot134 shot389 - shot
      ingredient278 ingredient430 ingredient305 - ingredient
      cocktail111 - cocktail
      dispenser28 dispenser94 dispenser453 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker235)
  (ontable shot134)
  (ontable shot389)
  (dispenses dispenser28 ingredient278)
  (dispenses dispenser94 ingredient430)
  (dispenses dispenser453 ingredient305)
  (clean shaker235)
  (clean shot134)
  (clean shot389)
  (empty shaker235)
  (empty shot134)
  (empty shot389)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker235 l0)
  (shaker-level shaker235 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail111 ingredient278)
  (cocktail-part2 cocktail111 ingredient305)
)
 (:goal
  (and
      (contains shot134 cocktail111)
)))
