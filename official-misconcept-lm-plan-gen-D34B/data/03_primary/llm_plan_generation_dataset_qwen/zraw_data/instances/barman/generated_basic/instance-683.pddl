(define (problem prob)
 (:domain barman)
 (:objects 
      shaker157 - shaker
      left right - hand
      shot205 shot64 - shot
      ingredient79 ingredient288 - ingredient
      cocktail1 - cocktail
      dispenser345 dispenser308 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker157)
  (ontable shot205)
  (ontable shot64)
  (dispenses dispenser345 ingredient79)
  (dispenses dispenser308 ingredient288)
  (clean shaker157)
  (clean shot205)
  (clean shot64)
  (empty shaker157)
  (empty shot205)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker157 l0)
  (shaker-level shaker157 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient79)
  (cocktail-part2 cocktail1 ingredient288)
)
 (:goal
  (and
      (contains shot205 cocktail1)
)))
