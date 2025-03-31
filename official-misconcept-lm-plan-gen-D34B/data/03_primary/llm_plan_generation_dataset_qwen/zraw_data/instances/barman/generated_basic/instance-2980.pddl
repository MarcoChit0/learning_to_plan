(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot73 shot141 - shot
      ingredient17 ingredient456 ingredient213 ingredient212 - ingredient
      cocktail242 - cocktail
      dispenser263 dispenser243 dispenser478 dispenser127 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot73)
  (ontable shot141)
  (dispenses dispenser263 ingredient17)
  (dispenses dispenser243 ingredient456)
  (dispenses dispenser478 ingredient213)
  (dispenses dispenser127 ingredient212)
  (clean shaker24)
  (clean shot73)
  (clean shot141)
  (empty shaker24)
  (empty shot73)
  (empty shot141)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail242 ingredient456)
  (cocktail-part2 cocktail242 ingredient213)
)
 (:goal
  (and
      (contains shot73 cocktail242)
)))
