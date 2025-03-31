(define (problem prob)
 (:domain barman)
 (:objects 
      shaker141 - shaker
      left right - hand
      shot234 shot352 shot142 - shot
      ingredient47 ingredient348 ingredient106 ingredient36 - ingredient
      cocktail315 - cocktail
      dispenser313 dispenser467 dispenser52 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker141)
  (ontable shot234)
  (ontable shot352)
  (ontable shot142)
  (dispenses dispenser313 ingredient47)
  (dispenses dispenser467 ingredient348)
  (dispenses dispenser52 ingredient106)
  (dispenses dispenser290 ingredient36)
  (clean shaker141)
  (clean shot234)
  (clean shot352)
  (clean shot142)
  (empty shaker141)
  (empty shot234)
  (empty shot352)
  (empty shot142)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker141 l0)
  (shaker-level shaker141 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail315 ingredient106)
  (cocktail-part2 cocktail315 ingredient47)
)
 (:goal
  (and
      (contains shot234 cocktail315)
      (contains shot352 cocktail315)
)))
