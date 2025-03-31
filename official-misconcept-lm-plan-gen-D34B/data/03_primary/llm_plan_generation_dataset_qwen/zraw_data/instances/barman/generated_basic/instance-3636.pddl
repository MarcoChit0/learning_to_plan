(define (problem prob)
 (:domain barman)
 (:objects 
      shaker246 - shaker
      left right - hand
      shot364 - shot
      ingredient183 ingredient331 ingredient23 ingredient228 - ingredient
      cocktail154 - cocktail
      dispenser239 dispenser71 dispenser476 dispenser346 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker246)
  (ontable shot364)
  (dispenses dispenser239 ingredient183)
  (dispenses dispenser71 ingredient331)
  (dispenses dispenser476 ingredient23)
  (dispenses dispenser346 ingredient228)
  (clean shaker246)
  (clean shot364)
  (empty shaker246)
  (empty shot364)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker246 l0)
  (shaker-level shaker246 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail154 ingredient331)
  (cocktail-part2 cocktail154 ingredient183)
)
 (:goal
  (and
      (contains shot364 cocktail154)
)))
