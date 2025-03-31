(define (problem prob)
 (:domain barman)
 (:objects 
      shaker108 - shaker
      left right - hand
      shot104 shot337 - shot
      ingredient69 ingredient303 ingredient139 - ingredient
      cocktail452 - cocktail
      dispenser96 dispenser225 dispenser135 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker108)
  (ontable shot104)
  (ontable shot337)
  (dispenses dispenser96 ingredient69)
  (dispenses dispenser225 ingredient303)
  (dispenses dispenser135 ingredient139)
  (clean shaker108)
  (clean shot104)
  (clean shot337)
  (empty shaker108)
  (empty shot104)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker108 l0)
  (shaker-level shaker108 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail452 ingredient139)
  (cocktail-part2 cocktail452 ingredient303)
)
 (:goal
  (and
      (contains shot104 cocktail452)
)))
