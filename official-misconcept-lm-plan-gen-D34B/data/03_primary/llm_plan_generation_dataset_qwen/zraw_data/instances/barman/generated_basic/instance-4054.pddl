(define (problem prob)
 (:domain barman)
 (:objects 
      shaker319 - shaker
      left right - hand
      shot186 shot436 - shot
      ingredient355 ingredient107 ingredient390 ingredient316 - ingredient
      cocktail2 - cocktail
      dispenser196 dispenser350 dispenser301 dispenser247 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker319)
  (ontable shot186)
  (ontable shot436)
  (dispenses dispenser196 ingredient355)
  (dispenses dispenser350 ingredient107)
  (dispenses dispenser301 ingredient390)
  (dispenses dispenser247 ingredient316)
  (clean shaker319)
  (clean shot186)
  (clean shot436)
  (empty shaker319)
  (empty shot186)
  (empty shot436)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker319 l0)
  (shaker-level shaker319 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail2 ingredient390)
  (cocktail-part2 cocktail2 ingredient355)
)
 (:goal
  (and
      (contains shot186 cocktail2)
)))
