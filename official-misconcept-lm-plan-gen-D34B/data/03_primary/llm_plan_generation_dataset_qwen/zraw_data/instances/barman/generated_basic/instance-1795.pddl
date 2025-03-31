(define (problem prob)
 (:domain barman)
 (:objects 
      shaker435 - shaker
      left right - hand
      shot10 shot149 - shot
      ingredient398 ingredient402 ingredient268 ingredient462 - ingredient
      cocktail187 - cocktail
      dispenser125 dispenser0 dispenser248 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker435)
  (ontable shot10)
  (ontable shot149)
  (dispenses dispenser125 ingredient398)
  (dispenses dispenser0 ingredient402)
  (dispenses dispenser248 ingredient268)
  (dispenses dispenser162 ingredient462)
  (clean shaker435)
  (clean shot10)
  (clean shot149)
  (empty shaker435)
  (empty shot10)
  (empty shot149)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker435 l0)
  (shaker-level shaker435 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail187 ingredient462)
  (cocktail-part2 cocktail187 ingredient398)
)
 (:goal
  (and
      (contains shot10 cocktail187)
)))
