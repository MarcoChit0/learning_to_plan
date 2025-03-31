(define (problem prob)
 (:domain barman)
 (:objects 
      shaker166 - shaker
      left right - hand
      shot49 shot36 shot231 - shot
      ingredient385 ingredient1 ingredient24 ingredient265 - ingredient
      cocktail1 - cocktail
      dispenser258 dispenser176 dispenser417 dispenser38 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker166)
  (ontable shot49)
  (ontable shot36)
  (ontable shot231)
  (dispenses dispenser258 ingredient385)
  (dispenses dispenser176 ingredient1)
  (dispenses dispenser417 ingredient24)
  (dispenses dispenser38 ingredient265)
  (clean shaker166)
  (clean shot49)
  (clean shot36)
  (clean shot231)
  (empty shaker166)
  (empty shot49)
  (empty shot36)
  (empty shot231)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker166 l0)
  (shaker-level shaker166 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient1)
  (cocktail-part2 cocktail1 ingredient385)
)
 (:goal
  (and
      (contains shot49 cocktail1)
      (contains shot36 cocktail1)
)))
