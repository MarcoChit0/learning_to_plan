(define (problem prob)
 (:domain barman)
 (:objects 
      shaker304 - shaker
      left right - hand
      shot423 shot186 - shot
      ingredient425 ingredient361 ingredient304 ingredient168 - ingredient
      cocktail165 - cocktail
      dispenser445 dispenser459 dispenser285 dispenser132 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker304)
  (ontable shot423)
  (ontable shot186)
  (dispenses dispenser445 ingredient425)
  (dispenses dispenser459 ingredient361)
  (dispenses dispenser285 ingredient304)
  (dispenses dispenser132 ingredient168)
  (clean shaker304)
  (clean shot423)
  (clean shot186)
  (empty shaker304)
  (empty shot423)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker304 l0)
  (shaker-level shaker304 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail165 ingredient425)
  (cocktail-part2 cocktail165 ingredient304)
)
 (:goal
  (and
      (contains shot423 cocktail165)
)))
