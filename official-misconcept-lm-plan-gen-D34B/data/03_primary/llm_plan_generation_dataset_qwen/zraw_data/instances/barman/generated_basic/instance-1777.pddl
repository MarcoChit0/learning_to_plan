(define (problem prob)
 (:domain barman)
 (:objects 
      shaker288 - shaker
      left right - hand
      shot391 shot122 - shot
      ingredient209 ingredient388 ingredient123 - ingredient
      cocktail356 - cocktail
      dispenser446 dispenser5 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker288)
  (ontable shot391)
  (ontable shot122)
  (dispenses dispenser446 ingredient209)
  (dispenses dispenser5 ingredient388)
  (dispenses dispenser465 ingredient123)
  (clean shaker288)
  (clean shot391)
  (clean shot122)
  (empty shaker288)
  (empty shot391)
  (empty shot122)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker288 l0)
  (shaker-level shaker288 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail356 ingredient123)
  (cocktail-part2 cocktail356 ingredient209)
)
 (:goal
  (and
      (contains shot391 cocktail356)
)))
