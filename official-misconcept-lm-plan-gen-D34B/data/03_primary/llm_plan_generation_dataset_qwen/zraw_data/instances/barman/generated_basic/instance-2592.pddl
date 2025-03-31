(define (problem prob)
 (:domain barman)
 (:objects 
      shaker318 - shaker
      left right - hand
      shot336 shot454 - shot
      ingredient210 ingredient13 ingredient296 - ingredient
      cocktail104 - cocktail
      dispenser205 dispenser381 dispenser137 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker318)
  (ontable shot336)
  (ontable shot454)
  (dispenses dispenser205 ingredient210)
  (dispenses dispenser381 ingredient13)
  (dispenses dispenser137 ingredient296)
  (clean shaker318)
  (clean shot336)
  (clean shot454)
  (empty shaker318)
  (empty shot336)
  (empty shot454)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker318 l0)
  (shaker-level shaker318 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail104 ingredient296)
  (cocktail-part2 cocktail104 ingredient210)
)
 (:goal
  (and
      (contains shot336 cocktail104)
)))
