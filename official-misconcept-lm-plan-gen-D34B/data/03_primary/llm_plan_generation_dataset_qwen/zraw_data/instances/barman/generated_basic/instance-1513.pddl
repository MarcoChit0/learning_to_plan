(define (problem prob)
 (:domain barman)
 (:objects 
      shaker219 - shaker
      left right - hand
      shot367 shot84 - shot
      ingredient57 ingredient153 ingredient77 ingredient189 - ingredient
      cocktail142 - cocktail
      dispenser346 dispenser103 dispenser415 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker219)
  (ontable shot367)
  (ontable shot84)
  (dispenses dispenser346 ingredient57)
  (dispenses dispenser103 ingredient153)
  (dispenses dispenser415 ingredient77)
  (dispenses dispenser52 ingredient189)
  (clean shaker219)
  (clean shot367)
  (clean shot84)
  (empty shaker219)
  (empty shot367)
  (empty shot84)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker219 l0)
  (shaker-level shaker219 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail142 ingredient77)
  (cocktail-part2 cocktail142 ingredient57)
)
 (:goal
  (and
      (contains shot367 cocktail142)
)))
