(define (problem prob)
 (:domain barman)
 (:objects 
      shaker59 - shaker
      left right - hand
      shot314 shot341 - shot
      ingredient16 ingredient308 ingredient448 ingredient31 - ingredient
      cocktail269 - cocktail
      dispenser294 dispenser19 dispenser352 dispenser102 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker59)
  (ontable shot314)
  (ontable shot341)
  (dispenses dispenser294 ingredient16)
  (dispenses dispenser19 ingredient308)
  (dispenses dispenser352 ingredient448)
  (dispenses dispenser102 ingredient31)
  (clean shaker59)
  (clean shot314)
  (clean shot341)
  (empty shaker59)
  (empty shot314)
  (empty shot341)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker59 l0)
  (shaker-level shaker59 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail269 ingredient31)
  (cocktail-part2 cocktail269 ingredient308)
)
 (:goal
  (and
      (contains shot314 cocktail269)
)))
