(define (problem prob)
 (:domain barman)
 (:objects 
      shaker251 - shaker
      left right - hand
      shot367 - shot
      ingredient109 ingredient56 ingredient360 ingredient199 - ingredient
      cocktail136 - cocktail
      dispenser444 dispenser137 dispenser427 dispenser307 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker251)
  (ontable shot367)
  (dispenses dispenser444 ingredient109)
  (dispenses dispenser137 ingredient56)
  (dispenses dispenser427 ingredient360)
  (dispenses dispenser307 ingredient199)
  (clean shaker251)
  (clean shot367)
  (empty shaker251)
  (empty shot367)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker251 l0)
  (shaker-level shaker251 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail136 ingredient56)
  (cocktail-part2 cocktail136 ingredient360)
)
 (:goal
  (and
      (contains shot367 cocktail136)
)))
