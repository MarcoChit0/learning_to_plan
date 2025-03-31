(define (problem prob)
 (:domain barman)
 (:objects 
      shaker448 - shaker
      left right - hand
      shot186 shot384 - shot
      ingredient487 ingredient217 - ingredient
      cocktail1 - cocktail
      dispenser226 dispenser232 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker448)
  (ontable shot186)
  (ontable shot384)
  (dispenses dispenser226 ingredient487)
  (dispenses dispenser232 ingredient217)
  (clean shaker448)
  (clean shot186)
  (clean shot384)
  (empty shaker448)
  (empty shot186)
  (empty shot384)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker448 l0)
  (shaker-level shaker448 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient217)
  (cocktail-part2 cocktail1 ingredient487)
)
 (:goal
  (and
      (contains shot186 cocktail1)
)))
