(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot316 shot155 - shot
      ingredient467 ingredient337 ingredient19 - ingredient
      cocktail268 - cocktail
      dispenser305 dispenser10 dispenser220 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot316)
  (ontable shot155)
  (dispenses dispenser305 ingredient467)
  (dispenses dispenser10 ingredient337)
  (dispenses dispenser220 ingredient19)
  (clean shaker262)
  (clean shot316)
  (clean shot155)
  (empty shaker262)
  (empty shot316)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail268 ingredient19)
  (cocktail-part2 cocktail268 ingredient337)
)
 (:goal
  (and
      (contains shot316 cocktail268)
)))
