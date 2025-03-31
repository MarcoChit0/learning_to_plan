(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot158 shot343 shot58 - shot
      ingredient115 ingredient365 - ingredient
      cocktail128 - cocktail
      dispenser492 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot158)
  (ontable shot343)
  (ontable shot58)
  (dispenses dispenser492 ingredient115)
  (dispenses dispenser306 ingredient365)
  (clean shaker42)
  (clean shot158)
  (clean shot343)
  (clean shot58)
  (empty shaker42)
  (empty shot158)
  (empty shot343)
  (empty shot58)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail128 ingredient365)
  (cocktail-part2 cocktail128 ingredient115)
)
 (:goal
  (and
      (contains shot158 cocktail128)
      (contains shot343 cocktail128)
)))
