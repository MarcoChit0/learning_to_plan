(define (problem prob)
 (:domain barman)
 (:objects 
      shaker144 - shaker
      left right - hand
      shot345 shot29 shot8 - shot
      ingredient113 ingredient120 ingredient446 ingredient380 - ingredient
      cocktail132 - cocktail
      dispenser334 dispenser350 dispenser16 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker144)
  (ontable shot345)
  (ontable shot29)
  (ontable shot8)
  (dispenses dispenser334 ingredient113)
  (dispenses dispenser350 ingredient120)
  (dispenses dispenser16 ingredient446)
  (dispenses dispenser191 ingredient380)
  (clean shaker144)
  (clean shot345)
  (clean shot29)
  (clean shot8)
  (empty shaker144)
  (empty shot345)
  (empty shot29)
  (empty shot8)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker144 l0)
  (shaker-level shaker144 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail132 ingredient446)
  (cocktail-part2 cocktail132 ingredient120)
)
 (:goal
  (and
      (contains shot345 cocktail132)
      (contains shot29 ingredient380)
)))
