(define (problem prob)
 (:domain barman)
 (:objects 
      shaker76 - shaker
      left right - hand
      shot99 shot246 shot118 - shot
      ingredient376 ingredient441 ingredient446 ingredient154 - ingredient
      cocktail302 - cocktail
      dispenser141 dispenser181 dispenser189 dispenser298 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker76)
  (ontable shot99)
  (ontable shot246)
  (ontable shot118)
  (dispenses dispenser141 ingredient376)
  (dispenses dispenser181 ingredient441)
  (dispenses dispenser189 ingredient446)
  (dispenses dispenser298 ingredient154)
  (clean shaker76)
  (clean shot99)
  (clean shot246)
  (clean shot118)
  (empty shaker76)
  (empty shot99)
  (empty shot246)
  (empty shot118)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker76 l0)
  (shaker-level shaker76 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail302 ingredient154)
  (cocktail-part2 cocktail302 ingredient446)
)
 (:goal
  (and
      (contains shot99 cocktail302)
      (contains shot246 cocktail302)
)))
