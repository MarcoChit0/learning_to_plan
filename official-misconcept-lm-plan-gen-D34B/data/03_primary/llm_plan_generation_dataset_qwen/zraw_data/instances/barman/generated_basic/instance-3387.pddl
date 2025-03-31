(define (problem prob)
 (:domain barman)
 (:objects 
      shaker455 - shaker
      left right - hand
      shot414 shot122 shot450 - shot
      ingredient404 ingredient63 ingredient434 - ingredient
      cocktail356 - cocktail
      dispenser133 dispenser106 dispenser29 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker455)
  (ontable shot414)
  (ontable shot122)
  (ontable shot450)
  (dispenses dispenser133 ingredient404)
  (dispenses dispenser106 ingredient63)
  (dispenses dispenser29 ingredient434)
  (clean shaker455)
  (clean shot414)
  (clean shot122)
  (clean shot450)
  (empty shaker455)
  (empty shot414)
  (empty shot122)
  (empty shot450)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker455 l0)
  (shaker-level shaker455 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail356 ingredient404)
  (cocktail-part2 cocktail356 ingredient434)
)
 (:goal
  (and
      (contains shot414 cocktail356)
      (contains shot122 cocktail356)
)))
