(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot486 shot105 shot365 - shot
      ingredient213 ingredient228 - ingredient
      cocktail255 - cocktail
      dispenser174 dispenser356 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot486)
  (ontable shot105)
  (ontable shot365)
  (dispenses dispenser174 ingredient213)
  (dispenses dispenser356 ingredient228)
  (clean shaker15)
  (clean shot486)
  (clean shot105)
  (clean shot365)
  (empty shaker15)
  (empty shot486)
  (empty shot105)
  (empty shot365)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail255 ingredient213)
  (cocktail-part2 cocktail255 ingredient228)
)
 (:goal
  (and
      (contains shot486 cocktail255)
      (contains shot105 ingredient228)
)))
