(define (problem prob)
 (:domain barman)
 (:objects 
      shaker311 - shaker
      left right - hand
      shot304 shot301 shot440 - shot
      ingredient217 ingredient337 ingredient472 ingredient177 - ingredient
      cocktail206 - cocktail
      dispenser298 dispenser210 dispenser461 dispenser356 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker311)
  (ontable shot304)
  (ontable shot301)
  (ontable shot440)
  (dispenses dispenser298 ingredient217)
  (dispenses dispenser210 ingredient337)
  (dispenses dispenser461 ingredient472)
  (dispenses dispenser356 ingredient177)
  (clean shaker311)
  (clean shot304)
  (clean shot301)
  (clean shot440)
  (empty shaker311)
  (empty shot304)
  (empty shot301)
  (empty shot440)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker311 l0)
  (shaker-level shaker311 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail206 ingredient177)
  (cocktail-part2 cocktail206 ingredient217)
)
 (:goal
  (and
      (contains shot304 cocktail206)
      (contains shot301 cocktail206)
)))
