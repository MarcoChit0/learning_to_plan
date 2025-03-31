(define (problem prob)
 (:domain barman)
 (:objects 
      shaker201 - shaker
      left right - hand
      shot63 shot385 - shot
      ingredient278 ingredient135 ingredient279 - ingredient
      cocktail485 - cocktail
      dispenser76 dispenser494 dispenser186 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker201)
  (ontable shot63)
  (ontable shot385)
  (dispenses dispenser76 ingredient278)
  (dispenses dispenser494 ingredient135)
  (dispenses dispenser186 ingredient279)
  (clean shaker201)
  (clean shot63)
  (clean shot385)
  (empty shaker201)
  (empty shot63)
  (empty shot385)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker201 l0)
  (shaker-level shaker201 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail485 ingredient135)
  (cocktail-part2 cocktail485 ingredient278)
)
 (:goal
  (and
      (contains shot63 cocktail485)
)))
