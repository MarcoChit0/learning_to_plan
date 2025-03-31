(define (problem prob)
 (:domain barman)
 (:objects 
      shaker36 - shaker
      left right - hand
      shot260 - shot
      ingredient488 ingredient154 ingredient242 ingredient65 - ingredient
      cocktail476 - cocktail
      dispenser301 dispenser154 dispenser278 dispenser404 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker36)
  (ontable shot260)
  (dispenses dispenser301 ingredient488)
  (dispenses dispenser154 ingredient154)
  (dispenses dispenser278 ingredient242)
  (dispenses dispenser404 ingredient65)
  (clean shaker36)
  (clean shot260)
  (empty shaker36)
  (empty shot260)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker36 l0)
  (shaker-level shaker36 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail476 ingredient242)
  (cocktail-part2 cocktail476 ingredient154)
)
 (:goal
  (and
      (contains shot260 cocktail476)
)))
