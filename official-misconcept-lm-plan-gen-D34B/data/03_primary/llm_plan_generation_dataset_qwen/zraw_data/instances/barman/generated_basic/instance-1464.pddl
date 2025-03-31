(define (problem prob)
 (:domain barman)
 (:objects 
      shaker152 - shaker
      left right - hand
      shot166 shot85 shot274 - shot
      ingredient108 ingredient221 - ingredient
      cocktail300 - cocktail
      dispenser21 dispenser157 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker152)
  (ontable shot166)
  (ontable shot85)
  (ontable shot274)
  (dispenses dispenser21 ingredient108)
  (dispenses dispenser157 ingredient221)
  (clean shaker152)
  (clean shot166)
  (clean shot85)
  (clean shot274)
  (empty shaker152)
  (empty shot166)
  (empty shot85)
  (empty shot274)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker152 l0)
  (shaker-level shaker152 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail300 ingredient221)
  (cocktail-part2 cocktail300 ingredient108)
)
 (:goal
  (and
      (contains shot166 cocktail300)
      (contains shot85 cocktail300)
)))
