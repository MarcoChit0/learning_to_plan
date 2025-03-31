(define (problem prob)
 (:domain barman)
 (:objects 
      shaker313 - shaker
      left right - hand
      shot425 shot125 - shot
      ingredient129 ingredient105 ingredient87 - ingredient
      cocktail201 - cocktail
      dispenser334 dispenser164 dispenser54 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker313)
  (ontable shot425)
  (ontable shot125)
  (dispenses dispenser334 ingredient129)
  (dispenses dispenser164 ingredient105)
  (dispenses dispenser54 ingredient87)
  (clean shaker313)
  (clean shot425)
  (clean shot125)
  (empty shaker313)
  (empty shot425)
  (empty shot125)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker313 l0)
  (shaker-level shaker313 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail201 ingredient105)
  (cocktail-part2 cocktail201 ingredient87)
)
 (:goal
  (and
      (contains shot425 cocktail201)
)))
