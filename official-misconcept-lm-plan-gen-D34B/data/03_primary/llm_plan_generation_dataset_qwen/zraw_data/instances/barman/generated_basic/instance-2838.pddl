(define (problem prob)
 (:domain barman)
 (:objects 
      shaker307 - shaker
      left right - hand
      shot132 shot410 - shot
      ingredient10 ingredient423 ingredient115 ingredient169 - ingredient
      cocktail87 - cocktail
      dispenser206 dispenser424 dispenser349 dispenser321 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker307)
  (ontable shot132)
  (ontable shot410)
  (dispenses dispenser206 ingredient10)
  (dispenses dispenser424 ingredient423)
  (dispenses dispenser349 ingredient115)
  (dispenses dispenser321 ingredient169)
  (clean shaker307)
  (clean shot132)
  (clean shot410)
  (empty shaker307)
  (empty shot132)
  (empty shot410)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker307 l0)
  (shaker-level shaker307 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail87 ingredient115)
  (cocktail-part2 cocktail87 ingredient169)
)
 (:goal
  (and
      (contains shot132 cocktail87)
)))
