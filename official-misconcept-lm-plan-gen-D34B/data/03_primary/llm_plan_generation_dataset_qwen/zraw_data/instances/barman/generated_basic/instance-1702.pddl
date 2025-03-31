(define (problem prob)
 (:domain barman)
 (:objects 
      shaker298 - shaker
      left right - hand
      shot367 - shot
      ingredient346 ingredient480 ingredient25 ingredient48 - ingredient
      cocktail304 - cocktail
      dispenser200 dispenser52 dispenser73 dispenser345 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker298)
  (ontable shot367)
  (dispenses dispenser200 ingredient346)
  (dispenses dispenser52 ingredient480)
  (dispenses dispenser73 ingredient25)
  (dispenses dispenser345 ingredient48)
  (clean shaker298)
  (clean shot367)
  (empty shaker298)
  (empty shot367)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker298 l0)
  (shaker-level shaker298 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail304 ingredient25)
  (cocktail-part2 cocktail304 ingredient480)
)
 (:goal
  (and
      (contains shot367 cocktail304)
)))
