(define (problem prob)
 (:domain barman)
 (:objects 
      shaker157 - shaker
      left right - hand
      shot168 - shot
      ingredient36 ingredient483 ingredient183 - ingredient
      cocktail131 - cocktail
      dispenser164 dispenser354 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker157)
  (ontable shot168)
  (dispenses dispenser164 ingredient36)
  (dispenses dispenser354 ingredient483)
  (dispenses dispenser200 ingredient183)
  (clean shaker157)
  (clean shot168)
  (empty shaker157)
  (empty shot168)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker157 l0)
  (shaker-level shaker157 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail131 ingredient36)
  (cocktail-part2 cocktail131 ingredient183)
)
 (:goal
  (and
      (contains shot168 cocktail131)
)))
