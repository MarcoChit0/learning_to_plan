(define (problem prob)
 (:domain barman)
 (:objects 
      shaker219 - shaker
      left right - hand
      shot149 shot322 - shot
      ingredient131 ingredient399 ingredient107 - ingredient
      cocktail154 - cocktail
      dispenser248 dispenser217 dispenser142 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker219)
  (ontable shot149)
  (ontable shot322)
  (dispenses dispenser248 ingredient131)
  (dispenses dispenser217 ingredient399)
  (dispenses dispenser142 ingredient107)
  (clean shaker219)
  (clean shot149)
  (clean shot322)
  (empty shaker219)
  (empty shot149)
  (empty shot322)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker219 l0)
  (shaker-level shaker219 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail154 ingredient131)
  (cocktail-part2 cocktail154 ingredient107)
)
 (:goal
  (and
      (contains shot149 cocktail154)
)))
