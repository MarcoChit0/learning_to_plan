(define (problem prob)
 (:domain barman)
 (:objects 
      shaker447 - shaker
      left right - hand
      shot180 shot298 shot428 - shot
      ingredient432 ingredient328 ingredient441 ingredient82 - ingredient
      cocktail97 - cocktail
      dispenser27 dispenser212 dispenser302 dispenser15 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker447)
  (ontable shot180)
  (ontable shot298)
  (ontable shot428)
  (dispenses dispenser27 ingredient432)
  (dispenses dispenser212 ingredient328)
  (dispenses dispenser302 ingredient441)
  (dispenses dispenser15 ingredient82)
  (clean shaker447)
  (clean shot180)
  (clean shot298)
  (clean shot428)
  (empty shaker447)
  (empty shot180)
  (empty shot298)
  (empty shot428)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker447 l0)
  (shaker-level shaker447 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail97 ingredient432)
  (cocktail-part2 cocktail97 ingredient441)
)
 (:goal
  (and
      (contains shot180 cocktail97)
      (contains shot298 cocktail97)
)))
