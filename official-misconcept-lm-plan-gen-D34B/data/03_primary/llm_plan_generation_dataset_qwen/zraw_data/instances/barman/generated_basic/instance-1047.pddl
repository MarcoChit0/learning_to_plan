(define (problem prob)
 (:domain barman)
 (:objects 
      shaker90 - shaker
      left right - hand
      shot426 shot365 shot398 - shot
      ingredient158 ingredient187 ingredient38 ingredient421 - ingredient
      cocktail1 - cocktail
      dispenser367 dispenser20 dispenser177 dispenser400 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker90)
  (ontable shot426)
  (ontable shot365)
  (ontable shot398)
  (dispenses dispenser367 ingredient158)
  (dispenses dispenser20 ingredient187)
  (dispenses dispenser177 ingredient38)
  (dispenses dispenser400 ingredient421)
  (clean shaker90)
  (clean shot426)
  (clean shot365)
  (clean shot398)
  (empty shaker90)
  (empty shot426)
  (empty shot365)
  (empty shot398)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker90 l0)
  (shaker-level shaker90 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient158)
  (cocktail-part2 cocktail1 ingredient421)
)
 (:goal
  (and
      (contains shot426 cocktail1)
      (contains shot365 ingredient158)
)))
