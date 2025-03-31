(define (problem prob)
 (:domain barman)
 (:objects 
      shaker441 - shaker
      left right - hand
      shot20 shot186 shot496 - shot
      ingredient36 ingredient30 ingredient40 ingredient232 - ingredient
      cocktail99 - cocktail
      dispenser98 dispenser99 dispenser498 dispenser158 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker441)
  (ontable shot20)
  (ontable shot186)
  (ontable shot496)
  (dispenses dispenser98 ingredient36)
  (dispenses dispenser99 ingredient30)
  (dispenses dispenser498 ingredient40)
  (dispenses dispenser158 ingredient232)
  (clean shaker441)
  (clean shot20)
  (clean shot186)
  (clean shot496)
  (empty shaker441)
  (empty shot20)
  (empty shot186)
  (empty shot496)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker441 l0)
  (shaker-level shaker441 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail99 ingredient30)
  (cocktail-part2 cocktail99 ingredient36)
)
 (:goal
  (and
      (contains shot20 cocktail99)
      (contains shot186 cocktail99)
)))
