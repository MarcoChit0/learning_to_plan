(define (problem prob)
 (:domain barman)
 (:objects 
      shaker435 - shaker
      left right - hand
      shot37 shot166 - shot
      ingredient188 ingredient223 - ingredient
      cocktail311 - cocktail
      dispenser498 dispenser349 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker435)
  (ontable shot37)
  (ontable shot166)
  (dispenses dispenser498 ingredient188)
  (dispenses dispenser349 ingredient223)
  (clean shaker435)
  (clean shot37)
  (clean shot166)
  (empty shaker435)
  (empty shot37)
  (empty shot166)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker435 l0)
  (shaker-level shaker435 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail311 ingredient188)
  (cocktail-part2 cocktail311 ingredient223)
)
 (:goal
  (and
      (contains shot37 cocktail311)
)))
