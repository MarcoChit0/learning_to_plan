(define (problem prob)
 (:domain barman)
 (:objects 
      shaker278 - shaker
      left right - hand
      shot76 - shot
      ingredient83 ingredient211 ingredient164 - ingredient
      cocktail1 - cocktail
      dispenser27 dispenser415 dispenser309 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker278)
  (ontable shot76)
  (dispenses dispenser27 ingredient83)
  (dispenses dispenser415 ingredient211)
  (dispenses dispenser309 ingredient164)
  (clean shaker278)
  (clean shot76)
  (empty shaker278)
  (empty shot76)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker278 l0)
  (shaker-level shaker278 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient164)
  (cocktail-part2 cocktail1 ingredient83)
)
 (:goal
  (and
      (contains shot76 cocktail1)
)))
