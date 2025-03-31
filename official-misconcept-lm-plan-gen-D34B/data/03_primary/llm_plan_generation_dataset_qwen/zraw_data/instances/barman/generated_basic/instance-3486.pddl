(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot152 shot246 shot40 - shot
      ingredient10 ingredient88 ingredient67 - ingredient
      cocktail326 - cocktail
      dispenser235 dispenser411 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot152)
  (ontable shot246)
  (ontable shot40)
  (dispenses dispenser235 ingredient10)
  (dispenses dispenser411 ingredient88)
  (dispenses dispenser262 ingredient67)
  (clean shaker126)
  (clean shot152)
  (clean shot246)
  (clean shot40)
  (empty shaker126)
  (empty shot152)
  (empty shot246)
  (empty shot40)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail326 ingredient88)
  (cocktail-part2 cocktail326 ingredient10)
)
 (:goal
  (and
      (contains shot152 cocktail326)
      (contains shot246 ingredient67)
)))
