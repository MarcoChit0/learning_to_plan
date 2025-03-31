(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot481 shot186 - shot
      ingredient191 ingredient248 ingredient268 - ingredient
      cocktail444 - cocktail
      dispenser453 dispenser52 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot481)
  (ontable shot186)
  (dispenses dispenser453 ingredient191)
  (dispenses dispenser52 ingredient248)
  (dispenses dispenser440 ingredient268)
  (clean shaker474)
  (clean shot481)
  (clean shot186)
  (empty shaker474)
  (empty shot481)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail444 ingredient191)
  (cocktail-part2 cocktail444 ingredient248)
)
 (:goal
  (and
      (contains shot481 cocktail444)
)))
