(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot497 shot139 shot218 - shot
      ingredient312 ingredient412 ingredient111 - ingredient
      cocktail104 - cocktail
      dispenser80 dispenser298 dispenser112 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot497)
  (ontable shot139)
  (ontable shot218)
  (dispenses dispenser80 ingredient312)
  (dispenses dispenser298 ingredient412)
  (dispenses dispenser112 ingredient111)
  (clean shaker262)
  (clean shot497)
  (clean shot139)
  (clean shot218)
  (empty shaker262)
  (empty shot497)
  (empty shot139)
  (empty shot218)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail104 ingredient412)
  (cocktail-part2 cocktail104 ingredient312)
)
 (:goal
  (and
      (contains shot497 cocktail104)
      (contains shot139 ingredient412)
)))
