(define (problem prob)
 (:domain barman)
 (:objects 
      shaker477 - shaker
      left right - hand
      shot87 shot180 shot46 - shot
      ingredient122 ingredient289 ingredient437 ingredient26 - ingredient
      cocktail1 - cocktail
      dispenser162 dispenser54 dispenser240 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker477)
  (ontable shot87)
  (ontable shot180)
  (ontable shot46)
  (dispenses dispenser162 ingredient122)
  (dispenses dispenser54 ingredient289)
  (dispenses dispenser240 ingredient437)
  (dispenses dispenser32 ingredient26)
  (clean shaker477)
  (clean shot87)
  (clean shot180)
  (clean shot46)
  (empty shaker477)
  (empty shot87)
  (empty shot180)
  (empty shot46)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker477 l0)
  (shaker-level shaker477 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient122)
  (cocktail-part2 cocktail1 ingredient437)
)
 (:goal
  (and
      (contains shot87 cocktail1)
      (contains shot180 cocktail1)
)))
